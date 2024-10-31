"""by lyuwenyu
"""

# --------------------------------------------------- #
# Modifications have been made to the time dimension. #
# --------------------------------------------------- #


import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import torchvision

# from src.core import register


__all__ = ['RTDETRPostProcessor']


# @register
class RTDETRPostProcessor(nn.Module):
    # __share__ = ['num_classes', 'use_focal_loss', 'num_top_queries', 'remap_mscoco_category']
    
    def __init__(self, num_classes=80, use_focal_loss=True, num_top_queries=300, remap_mscoco_category=False, deploy_mode=False, use_score_threshold=False, score_threshold=1.0, use_nms=True, iou_threshold=0.5) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = num_classes
        self.remap_mscoco_category = remap_mscoco_category 
        self.deploy_mode = deploy_mode

        self.use_score_threshold = use_score_threshold
        self.score_threshold = score_threshold
        self.use_nms = use_nms
        self.iou_threshold = iou_threshold

    def extra_repr(self) -> str:
        return f'use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}'
    
    def forward(self, outputs, orig_target_sizes):

        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']
        # orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)        

        torch.save(boxes, "/home/catlab/py_code/gt4dvs/save_dir/processing/boxes_cxcywh_normed.pt")
        
        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')
        torch.save(bbox_pred, "/home/catlab/py_code/gt4dvs/save_dir/processing/boxes_xyxy_normed.pt")
        
        scale_img_w, img_h = orig_target_sizes[0]
        assert scale_img_w == img_h, "scale_img_w and img_h must be equal"
        
        bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)
        # bbox_pred = bbox_pred.clamp(min=0, max=img_h-1)
        torch.save(bbox_pred, "/home/catlab/py_code/gt4dvs/save_dir/processing/boxes_xyxy.pt")

        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            scores, index = torch.topk(scores.flatten(1), self.num_top_queries, axis=-1)
            labels = index % self.num_classes
            index = index // self.num_classes
            boxes = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))
            
        else:
            scores = F.softmax(logits)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            if scores.shape[1] > self.num_top_queries:
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                boxes = torch.gather(boxes, dim=1, index=index.unsqueeze(-1).tile(1, 1, boxes.shape[-1]))
        
        # TODO for onnx export
        if self.deploy_mode:
            return labels, boxes, scores

        # # TODO
        # if self.remap_mscoco_category:
        #     from ...data.coco import mscoco_label2category
        #     labels = torch.tensor([mscoco_label2category[int(x.item())] for x in labels.flatten()])\
        #         .to(boxes.device).reshape(labels.shape)

        results = []
        for lab, box, sco in zip(labels, boxes, scores):
            if self.use_score_threshold:
                score_threshold = torch.mean(sco) if self.score_threshold==1.0 else self.score_threshold
                mask = sco > score_threshold
                flt_lab = lab[mask]
                flt_box = box[mask]
                flt_sco = sco[mask]
            else:
                flt_lab = lab
                flt_box = box
                flt_sco = sco

            if self.use_nms:
                nms_indices = torchvision.ops.nms(flt_box, flt_sco, iou_threshold=self.iou_threshold)
                flt_lab = flt_lab[nms_indices]
                flt_box = flt_box[nms_indices]
                flt_sco = flt_sco[nms_indices]

            result = dict(labels=flt_lab, boxes=flt_box, scores=flt_sco)
            results.append(result)
        torch.save(results, "/home/catlab/py_code/gt4dvs/save_dir/processing/results.pt")
        
        return results


    def deploy(self, ):
        self.eval()
        self.deploy_mode = True
        return self 

    @property
    def iou_types(self, ):
        return ('bbox', )
