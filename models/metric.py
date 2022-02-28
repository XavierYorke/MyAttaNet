import torch
from metric.segmentation import SegmentationMetric
mertric = SegmentationMetric(2)


def demo_metric(outputs, target):
    with torch.no_grad():
        pred = torch.argmax(outputs[0], dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def dice_metric(outputs, target):
    with torch.no_grad():
        batch_size = target.size(0)
        reslut_dice = 0.0
        beta = 0.5
        target = target.permute(0, 2, 3, 1)

        outputs = outputs[0].permute(0, 2, 3, 1)
        for i in range(batch_size):

            prob = outputs[i]
            ref = target[i]

            alpha = 1.0 - beta
            # TP ：ref * prob 两边都是positive
            # FP  ：(1 - ref) * prob 负的标签 正的预测
            # TN ：两边都是负的
            # FN ：ref*(1-prob）预测是负的
            tp = (ref * prob).sum()  # 真阳
            fp = ((1 - ref) * prob).sum()  # 假阳
            fn = (ref * (1 - prob)).sum()  # 假阴
            # alpha beta 分别控制FP 和 FN的惩罚度
            dice = tp / (tp + alpha * fp + beta * fn)
            reslut_dice = reslut_dice + dice
    return reslut_dice / batch_size


def pa(outputs, target):
    with torch.no_grad():
        metric = SegmentationMetric(2)  # 3表示有3个分类，有几个分类就填几
        outputs = torch.where(outputs[0] > 0.5, torch.ones_like(
            outputs[0]), torch.zeros_like(outputs[0]))
        # print(target.shape)
        outputs = outputs.permute(0, 2, 3, 1)
        target = target.permute(0, 2, 3, 1)
        metric.addBatch(outputs, target)
        pa = metric.pixelAccuracy()
    return pa


def cpa(outputs, target):
    with torch.no_grad():
        metric = SegmentationMetric(2)  # 3表示有3个分类，有几个分类就填几
        outputs = torch.where(outputs[0] > 0.5, torch.ones_like(
            outputs[0]), torch.zeros_like(outputs[0]))
        outputs = outputs.permute(0, 2, 3, 1)
        target = target.permute(0, 2, 3, 1)
        metric.addBatch(outputs, target)
        cpa = metric.classPixelAccuracy()
    return cpa


def mpa(outputs, target):
    with torch.no_grad():
        metric = SegmentationMetric(2)  # 3表示有3个分类，有几个分类就填几
        outputs = torch.where(outputs[0] > 0.5, torch.ones_like(
            outputs[0]), torch.zeros_like(outputs[0]))
        outputs = outputs.permute(0, 2, 3, 1)
        target = target.permute(0, 2, 3, 1)
        metric.addBatch(outputs, target)
        mpa = metric.meanPixelAccuracy()
    return mpa


def mIou(outputs, target):
    with torch.no_grad():
        metric = SegmentationMetric(2)  # 3表示有3个分类，有几个分类就填几
        outputs = torch.where(outputs[0] > 0.5, torch.ones_like(
            outputs[0]), torch.zeros_like(outputs[0]))
        outputs = outputs.permute(0, 2, 3, 1)
        target = target.permute(0, 2, 3, 1)
        metric.addBatch(outputs, target)
        mIoU = metric.meanIntersectionOverUnion()
    return mIoU
