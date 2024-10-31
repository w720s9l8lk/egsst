from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

try:
    from torch import compile as th_compile
except ImportError:
    th_compile = None

from ..build.model_build import ModelBuilder
from ..utils.timers import TimerDummy as CudaTimer


class GraphTransDetector(nn.Module):
    def __init__(self, yaml_file_path:str):
        super().__init__()
        self.model_builder = ModelBuilder(yaml_file_path)
        
        self.g_backbone = self.model_builder.build_g_backbone()
        self.s_backbone = self.model_builder.build_tr_backbone()

        self.rtdetr_head = self.model_builder.build_rtdetr_head()

    def forward_backbone(self, ev_data):
        # merged_gcn_matrix, space_prob, max_time_matrix_tensor = self.g_backbone(non_overlap_data, overlap_data) # [B, T, C, X, Y] and [B, 1] and [B, T, X, Y, 1]
        batch_gcn_matrices, space_prob = self.g_backbone(ev_data) # [B*T, C, Y=H, Y=W] and [1]
        
        # * space_prob: a value
        # space_prob_mean = space_prob
        # * batch_gcn_matrices: (B*T, C, H, W)
        # batch_gcn_matrices = merged_gcn_matrix.reshape(-1, *merged_gcn_matrix.shape[2:]) # (B*T, C, X, Y)
        
        # if space_prob_mean >=0.6: # space_trans
        #     # print("第一种情况 全s")
        #     features_234_dict, _ = self.s_backbone(batch_gcn_matrices)
        # elif space_prob_mean <= 0.3: # time_trans
        #     # print("第二种情况 全t")
        #     batch_gcn_matrices_with_st = torch.cat((batch_gcn_matrices, self.init_st), dim=0) # (B*T+1, C, X, Y)
        #     features_234_dict, _ = self.t_backbone(batch_gcn_matrices_with_st, max_time_matrix_tensor)
        # else: # space_trans + time_trans
        #     # print("第三种情况 s+t")
        #     features_234_dict_s, batch_feature_st = self.s_backbone(batch_gcn_matrices) # batch_feature_st: (B*T, C, X, Y)
        #     last_feature_st = batch_feature_st[-1, ...].unsqueeze(0) # (1, C, X, Y)
        #     batch_gcn_matrices_with_st = torch.cat((batch_gcn_matrices, last_feature_st), dim=0) # (B*T+1, C, X, Y)
        #     features_234_dict_t, _ = self.t_backbone(batch_gcn_matrices_with_st, max_time_matrix_tensor)
        #     features_234_dict = {k: space_prob_mean*features_234_dict_s[k] + (1-space_prob_mean)*features_234_dict_t[k] for k in features_234_dict_s}
        
        # if space_prob_mean >=0.4: # space_trans
        #     print("第一种情况 全s")
        #     features_234_dict, _ = self.s_backbone(batch_gcn_matrices)
        # else: #  space_prob_mean < 0.6: # time_trans
        #     print("第二种情况 全t")
        #     batch_gcn_matrices_with_st = torch.cat((batch_gcn_matrices, self.init_st), dim=0) # (B*T+1, C, X, Y)
        #     features_234_dict, _ = self.t_backbone(batch_gcn_matrices_with_st, max_time_matrix_tensor)

        # only s_trans
        features_234_dict = self.s_backbone(batch_gcn_matrices)
        # (B*T, C, Y=H, Y=W)

        # # only t_trans
        # batch_gcn_matrices_with_st = torch.cat((batch_gcn_matrices, self.init_st), dim=0) # (B*T+1, C, X, Y)
        # features_234_dict, _ = self.t_backbone(batch_gcn_matrices_with_st, max_time_matrix_tensor)
            
        # # # s_trans + t_trans
        # features_234_dict_s, batch_feature_st = self.s_backbone(batch_gcn_matrices) # batch_feature_st: (B*T, C, X, Y)
        # last_feature_st = batch_feature_st[-1, ...].unsqueeze(0) # (1, C, X, Y)
        # batch_gcn_matrices_with_st = torch.cat((batch_gcn_matrices, last_feature_st), dim=0) # (B*T+1, C, X, Y)
        # features_234_dict_t, _ = self.t_backbone(batch_gcn_matrices_with_st, max_time_matrix_tensor)
        # features_234_dict = {k: space_prob_mean*features_234_dict_s[k] + (1-space_prob_mean)*features_234_dict_t[k] for k in features_234_dict_s}

        torch.save(features_234_dict, "/home/catlab/py_code/gt4dvs/save_dir/processing/features_234_dict.pt")
        return features_234_dict

    def forward_detect(self,
                       backbone_features, # Dict[int, torch.Tensor]
                       targets: Optional[torch.Tensor] = None
                       ):
        return self.rtdetr_head(backbone_features, targets)


    def forward(self,
                ev_BTN4, 
                targets: Optional[torch.Tensor] = None):
        features_dict = self.forward_backbone(ev_BTN4)
        return self.rtdetr_head(features_dict, targets)
    


