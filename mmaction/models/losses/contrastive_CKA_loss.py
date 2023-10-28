import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import LOSSES
from mmcv.runner import get_dist_info
from mmaction.core.hooks.fp16_utils import force_fp32
from mmaction.models.utils.gather_loss import GatherLoss, VariedShapeGatherLoss
from .loss_utils import *

@LOSSES.register_module()
class NormSoftmaxLossWithCKA(nn.Module):
    def __init__(self, temperature=0.07, lambdaCKA=0.1, cos_sim=False):
        super().__init__()
        self.t = temperature
        self.lambdaCKA = lambdaCKA
        self.use_cos_similarity = cos_sim
        self.allgather = GatherLoss.apply
        self.rank, self.world_size = get_dist_info()
        if self.use_cos_similarity:
            print("use cosine similarity")
        self.fp16_enabled = False

    @force_fp32()
    def forward(self, video_embd=None, text_embd=None, sim_mat=None):
        CKAloss = torch.tensor(0.0)
        
        if sim_mat is None:           
            video_embd = self.allgather(video_embd, self.rank, self.world_size) # video_embd shape: B x D
            text_embd = self.allgather(text_embd, self.rank, self.world_size) # text_embd  shape: B x D

            # calculate CKAloss
            CKAloss = torch.tensor(CKA(video_embd, text_embd))
            
            if self.use_cos_similarity:
                x = sim_matrix(video_embd, text_embd) / self.t
            else:
                video_embd = F.normalize(video_embd, dim=-1)
                text_embd = F.normalize(text_embd, dim=-1)
                x = torch.matmul(video_embd, text_embd.t()) / self.t
            "Assumes input x is similarity matrix of N x M \in [-1, 1], computed using the cosine similarity between normalised vectors"
        else:
            x = sim_mat
        
        i_logsm = F.log_softmax(x, dim=1)
        j_logsm = F.log_softmax(x.t(), dim=1)

        # sum over positives
        idiag = torch.diag(i_logsm)
        loss_i = idiag.sum() / len(idiag)

        jdiag = torch.diag(j_logsm)
        loss_j = jdiag.sum() / len(jdiag)

        return - loss_i - loss_j - self.lambdaCKA * CKAloss



