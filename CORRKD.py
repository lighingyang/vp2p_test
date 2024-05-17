import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller
from mdistiller.engine.corr import ContrastiveCorrelationLoss
from mdistiller.engine.dimtrans import ChannelTransformer


def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    # log_pred_teacher = torch.log(pred_teacher)  # houmh
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    # tckd_loss_ST = (
    #     F.kl_div(log_pred_teacher, pred_student, size_average=False)
    #     * (temperature**2)
    #     / target.shape[0]
    # )
    # tckd_loss = (tckd_loss_ST + tckd_loss_TS) / 2


    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    # log_pred_teacher_part2 = F.log_softmax(
    #     logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    # ) # houmh
    # pred_student_part2 = F.softmax(
    #     logits_student / temperature - 1000.0 * gt_mask, dim=1
    # )   # houmh
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    # nckd_loss_ST= (
    #     F.kl_div(log_pred_teacher_part2, pred_student_part2, size_average=False)
    #     * (temperature**2)
    #     / target.shape[0]
    # )
    # nckd_loss = (nckd_loss_ST + nckd_loss_TS) / 2
    return alpha * tckd_loss + beta * nckd_loss


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt


class CORRKD(Distiller):
    

    def __init__(self, student, teacher, cfg):
        super(CORRKD, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.CORRKD.CE_WEIGHT
        self.alpha = cfg.CORRKD.ALPHA
        self.beta = cfg.CORRKD.BETA
        self.temperature = cfg.CORRKD.T
        self.warmup = cfg.CORRKD.WARMUP
        self.corr_begin_epoch = cfg.CORRKD.CORR_BEGIN_EPOCH
        self.tea_corr_1 = cfg.CORRKD.TEA_CORR_1
        self.stu_corr_1 = cfg.CORRKD.STU_CORR_1
        self.tea_corr_2 = cfg.CORRKD.TEA_CORR_2
        self.stu_corr_2 = cfg.CORRKD.STU_CORR_2

    def forward_train(self, image, target, **kwargs):
        logits_student, feats_student = self.student(image)

        # data enhancement branch
        with torch.no_grad():
            logits_teacher, feats_teacher = self.teacher(image)
        

        # losses
        # loss_ce_T = self.ce_loss_weight * F.cross_entropy(logits_teacher, target)
        loss_ce_S = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_dkd = min(kwargs["epoch"] / self.warmup, 1.0) * dkd_loss(
            logits_student,
            logits_teacher,
            target,
            self.alpha,
            self.beta,
            self.temperature,
        )

# ----------------------------------------------------------------------------------------------------------------------
        # Don't calculate loss_corr before epoch 150
        # print("epoch - loss_corr_houmh", self.corr_begin_epoch)
        # print("self.tea_corr_1", self.tea_corr_1)
        # print("self.stu_corr_1", self.stu_corr_1)
        # print("self.tea_corr_2", self.tea_corr_2)
        # print("self.stu_corr_2", self.stu_corr_2)

        # b = input()
        # kwargs["epoch"] = 80    # debug


        if kwargs["epoch"] >= self.corr_begin_epoch:
            loss_corr = ContrastiveCorrelationLoss()
            
            tea_corr_1 = feats_teacher["feats"][self.tea_corr_1]
            tea_corr_2 = feats_teacher["feats"][self.tea_corr_2]
            stu_corr_1 = feats_student["feats"][self.stu_corr_1]
            stu_corr_2 = feats_student["feats"][self.stu_corr_2]
            
            # t-s channel trans
            tea_c1 = tea_corr_1.size(1)
            stu_c1 = stu_corr_1.size(1)

            if tea_c1 != stu_c1:
                channel_trans = ChannelTransformer(max(tea_c1, stu_c1), min(tea_c1, stu_c1))
                
                if torch.cuda.is_available():
                    channel_trans = channel_trans.to('cuda')
                
                if tea_c1 > stu_c1:
                    tea_corr_1 = channel_trans(tea_corr_1)
                else:
                    stu_corr_1 = channel_trans(stu_corr_1)
                
            # print(f"Adjusted tea_corr_1 (channel {tea_c1} to {tea_corr_1.size(1)}):\n{tea_corr_1}")
            # print(f"Adjusted tea_corr_2 (channel {tea_c2} to {tea_corr_2.size(1)}):\n{tea_corr_2}")
            # stu_channel trans
            tea_c2 = tea_corr_2.size(1)
            stu_c2 = stu_corr_2.size(1)

            if tea_c2 != stu_c2:
                channel_trans = ChannelTransformer(max(tea_c2, stu_c2), min(tea_c2, stu_c2))

                if torch.cuda.is_available():
                    channel_trans = channel_trans.to('cuda')

                if tea_c2 > stu_c2:
                    tea_corr_2 = channel_trans(tea_corr_2)
                else:
                    stu_corr_2 = channel_trans(stu_corr_2)


            # T—T/S-S Corr
            loss_corr = loss_corr(
                tea_corr_1, stu_corr_1, 
                tea_corr_2, stu_corr_2
                )

            losses_dict = {
                "loss_ce_S": loss_ce_S,
                # "loss_ce_T": loss_ce_T,
                "loss_kd": loss_dkd,
                "loss_corr": loss_corr,
            }
        else:
            losses_dict = {
                "loss_ce_S": loss_ce_S,
                # "loss_ce_T": loss_ce_T,
                "loss_kd": loss_dkd,
                # "loss_corr": loss_corr,
            }# Set loss_corr to 0 before epoch 150

        return logits_student, losses_dict
