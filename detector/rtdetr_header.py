import torch
import torch.nn as nn
import numpy as np
import sys
import os
import contextlib
import math

from omegaconf import OmegaConf

from datasets.detector.rtdetr_head.rtdetr_converter import convert_yolo_batch_to_targets_format, move_to_device, to_feats_list, convert_yolo_batch_to_coco_format
from datasets.detector.rtdetr_head.rtdetr_hybrid_encoder import HybridEncoder
from datasets.detector.rtdetr_head.rtdetr_decoder import RTDETRTransformer
from datasets.detector.rtdetr_head.rtdetr_matcher import HungarianMatcher
from datasets.detector.rtdetr_head.rtdetr_criterion import SetCriterion
from datasets.detector.rtdetr_head.rtdetr_postprocessor import RTDETRPostProcessor
from datasets.detector.rtdetr_head.rtdetr_coco_eval import CocoEvaluator


class RTDETRHead(nn.Module):
    def __init__(self, yaml_file_path):
        super().__init__()
        self.cfg = OmegaConf.load(yaml_file_path)
        self.dataset_name = self.cfg.dataset.current_dataset
        cfg_rtdetr = self.cfg.rtdetr
        cfg_enc_dec_uniform = cfg_rtdetr.uniform
        cfg_hybrid_encoder = cfg_rtdetr.hybrid_encoder
        cfg_decoder = cfg_rtdetr.decoder
        cfg_matcher = cfg_rtdetr.matcher
        cfg_criterion = cfg_rtdetr.criterion
        cfg_postprocess = cfg_rtdetr.postprocess

        self.encoder = HybridEncoder(**cfg_enc_dec_uniform, **cfg_hybrid_encoder)
        self.decoder = RTDETRTransformer(**cfg_enc_dec_uniform, **cfg_decoder)
        self.matcher = HungarianMatcher(**cfg_matcher)
        self.criterion = SetCriterion(matcher=self.matcher, **cfg_criterion)
        self.postprocess = RTDETRPostProcessor(**cfg_postprocess)

        self.use_coco_eval = cfg_rtdetr.coco_eval.use_coco_eval
        

    def enc_dec_loss(self, feats_list, target_format_lbl=None):
        encoder_output = self.encoder(feats_list) # [8, 128, 20, 20] [8, 128, 10, 10] [8, 128, 5, 5]
        decoder_outputs = self.decoder(encoder_output, targets=target_format_lbl)
        '''
        decoder_outputs: 'pred_logits', 'pred_boxes', 'aux_outputs', 'dn_aux_outputs', 'dn_meta'
            # output[pred_logits].shape: torch.Size([4, 300, 2])
            # output[pred_boxes].shape: torch.Size([4, 300, 4])
        '''

        loss_dict = self.criterion(decoder_outputs, target_format_lbl)
        sum_loss = sum(loss_dict.values())
        
        # check if loss is finite (not NaN and not infinite)
        if not math.isfinite(sum_loss.item()):
            print(f"Loss is {sum_loss.item()}, stopping training")
            sys.exit(1)

        return decoder_outputs, sum_loss, loss_dict
    
    def coco_eval(self, tgt_tensor, post_results):
        assert self.training==False, "coco_eval should be used in eval mode"
        assert self.dataset_name in {'gen1', 'gen4'}
        classes = ("car", "pedestrian") if self.dataset_name=='gen1' else ("pedestrian", "two-wheeler", "car")
        scale_img_width = 240 if self.dataset_name=='gen1' else 720
        img_height = 240 if self.dataset_name=='gen1' else 720

        coco_targets = convert_yolo_batch_to_coco_format(tgt_tensor, img_width=scale_img_width, img_height=img_height, labelmap=classes)
        torch.save(coco_targets, "/home/catlab/py_code/gt4dvs/save_dir/processing/coco_targets.pt")
        # iou_types = ('bbox', )
        iou_types = self.postprocess.iou_types # ('bbox', )
        coco_evaluator = CocoEvaluator(coco_targets, iou_types)

        img_ids = coco_targets.getImgIds()
        res = {img_id: output for img_id, output in zip(img_ids, post_results)}
        for img_id in res:
            res[img_id]['labels'] += 1 # ******** from 0,1 to 1,2

        torch.save(res, "/home/catlab/py_code/gt4dvs/save_dir/processing/res.pt")
        
        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
            if coco_evaluator is not None:
                coco_evaluator.update(res)

            # gather the stats from all processes
                coco_evaluator.synchronize_between_processes()

            # accumulate predictions from all images
                coco_evaluator.accumulate()
                coco_evaluator.summarize()

        torch.save(coco_evaluator.coco_eval, "/home/catlab/py_code/gt4dvs/save_dir/processing/coco_evaluator_coco_eval.pt")
        # stats = {}
        # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        out_keys = ('AP', 'AP_50', 'AP_75', 'AP_S', 'AP_M', 'AP_L', 'AR_1', 'AR_10', 'AR_100', 'AR_S', 'AR_M', 'AR_L')
        out_dict = {k: 0.0000 for k in out_keys}  # 初始化字典，初始值设为五位小数的浮点数

        if hasattr(coco_evaluator, 'coco_eval') and 'bbox' in coco_evaluator.coco_eval:
            # 确保COCO评估器已经进行了评估并有结果
            for idx, key in enumerate(out_keys):
                # 格式化浮点数为字符串，保留五位小数，然后转换回浮点数存储
                formatted_value = format(coco_evaluator.coco_eval['bbox'].stats[idx], '.4f')
                out_dict[key] = float(formatted_value)

        return out_dict

    def forward(self, feats_dict, targets=None):
        scale_img_width = 240 if self.dataset_name=='gen1' else 720
        img_height = 240 if self.dataset_name=='gen1' else 720

        feats_list = to_feats_list(feats_dict)
        target_format_lbl = convert_yolo_batch_to_targets_format(targets, img_width=scale_img_width, img_height=img_height)
        target_format_lbl = move_to_device(target_format_lbl, feats_list[0].device)

        decoder_outputs, sum_loss, loss_dict = self.enc_dec_loss(feats_list, target_format_lbl)
        
        if self.training:
            return sum_loss, loss_dict
        
        if self.use_coco_eval and self.training==False:
            orig_target_sizes = torch.stack([t["orig_size"] for t in target_format_lbl], dim=0)
            torch.save(orig_target_sizes, "/home/catlab/py_code/gt4dvs/save_dir/processing/orig_target_sizes.pt")
            post_results = self.postprocess(decoder_outputs, orig_target_sizes) # xyxy
            out_dict = self.coco_eval(targets, post_results)
            return out_dict


