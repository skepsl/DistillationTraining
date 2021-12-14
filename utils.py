
import torch.nn as nn
from torch.nn import functional as F


def init_weight(layer):
    nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, 0)


class Distillation(nn.Module):
    def __init__(self, alpha=0.5, tau=0.5):
        super(Distillation, self).__init__()
        self.hard_loss = nn.CrossEntropyLoss()
        self.alpha = alpha
        self.tau = tau

    def forward(self, me1, me2, me3, le, gt):  # Multi-exit, Last Exit, Ground Truth
        T = self.tau
        hard_loss1 = self.hard_loss(me1, gt)
        hard_loss2 = self.hard_loss(me2, gt)
        hard_loss3 = self.hard_loss(me3, gt)
        hard_loss4 = self.hard_loss(le, gt)

        distillation_loss1 = F.kl_div(
            F.log_softmax(me1 / T, dim=1),
            F.log_softmax(le / T, dim=1),
            reduction='sum',
            log_target=True
        ) * (T * T) / me1.numel()

        distillation_loss2 = F.kl_div(
            F.log_softmax(me2 / T, dim=1),
            F.log_softmax(le / T, dim=1),
            reduction='sum',
            log_target=True
        ) * (T * T) / me2.numel()

        distillation_loss3 = F.kl_div(
            F.log_softmax(me3 / T, dim=1),
            F.log_softmax(le / T, dim=1),
            reduction='sum',
            log_target=True
        ) * (T * T) / me3.numel()

        loss1 = hard_loss1 * (1 - self.alpha) + distillation_loss1 * self.alpha
        loss2 = hard_loss2 * (1 - self.alpha) + distillation_loss2 * self.alpha
        loss3 = hard_loss3 * (1 - self.alpha) + distillation_loss3 * self.alpha
        loss4 = hard_loss4

        return loss1, loss2, loss3, loss4
