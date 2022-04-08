import pandas as pd
import os
import torch
from medpy import metric
import numpy as np


def train_step(model, images, labels):
    # 训练模式，dropout层发生作用
    model.train()

    # 梯度清零
    model.optimizer.zero_grad()
    # 正向传播求损失
    loss, _ = shared_step(model, images, labels)
    # 反向传播求梯度
    loss.backward()
    model.optimizer.step()
    return loss.item()


def valid_step(model, images, labels):
    # 预测模式，dropout层不发生作用
    model.eval()
    loss, outputs = shared_step(model, images, labels)
    dice, precision, recall, tnr = eval_step(outputs, labels)
    return loss.item(), dice, precision, recall, tnr


def shared_step(model, images, labels):
    out, out16, out32 = model(images)
    lossp = model.criteria_p(out, labels)
    loss1 = model.criteria_aux1(out16, labels)
    loss2 = model.criteria_aux2(out32, labels)
    loss = lossp + loss1 + loss2
    return loss, out
    # predictions = model(images)
    # loss = model.loss_func(predictions, labels)
    # return loss.item()


def eval_step(result, labels):
    dices, precisions, recalls, tnrs = [], [], [], []
    for index in range(result.shape[0]):
        image = torch.argmax(result[index], dim=0).cpu().detach().numpy()
        # image = result[index].cpu().detach().numpy().transpose(1, 2, 0)
        # image = np.squeeze(image, axis=-1)
        label = labels[index].cpu().detach().numpy()

        dice = metric.dc(image.astype(np.uint8), label.astype(np.uint8))
        precision = metric.precision(image.astype(np.uint8), label.astype(np.uint8))
        recall = metric.recall(image.astype(np.uint8), label.astype(np.uint8))
        tnr = metric.true_negative_rate(image.astype(np.uint8), label.astype(np.uint8))
        dices.append(dice)
        precisions.append(precision)
        recalls.append(recall)
        tnrs.append(tnr)
    dice = np.mean(dices)
    precision = np.mean(precisions)
    recall = np.mean(recalls)
    tnr = np.mean(tnrs)
    # msg = "dice: {:.4f} precision: {:.4f} recall: {:.4f} tnr: {:.4f}".format(dice, precision,  recall, tnr)
    return dice, precision, recall, tnr


def train(model, start, epochs, dl_train, dl_valid, log_step_freq, logger, device, output_dir):
    notes = pd.DataFrame(columns=["epoch", "loss", "val_loss"])
    logger.info("Start Training...")
    for epoch in range(start, epochs + 1):
        # 1，train-------------------------------------------------
        loss_sum = 0.0
        step = 1
        for step, (images, labels) in enumerate(dl_train, 1):
            images = images.to(device)
            labels = labels.to(device)
            
            loss = train_step(model, images, labels)
            loss_sum += loss
            # 打印batch级别日志
            # if step % log_step_freq == 0:
            #     msg = "[step = {}] loss: {:.3f}".format(step, loss_sum / step)
            #     logger.info(msg)

        # 2，val-------------------------------------------------
        val_loss_sum = 0.0
        val_step = 1
        dices, precisions, recalls, tnrs = [], [], [], []
        for val_step, (images, labels) in enumerate(dl_valid, 1):
            images = images.to(device)
            labels = labels.to(device)

            val_loss, dice, precision, recall, tnr = valid_step(model, images, labels)
            val_loss_sum += val_loss
            dices.append(dice)
            precisions.append(precision)
            recalls.append(recall)
            tnrs.append(tnr)
        dice = np.mean(dices)
        precision = np.mean(precisions)
        recall = np.mean(recalls)
        tnr = np.mean(tnrs)
        msg = "dice: {:.4f} precision: {:.4f} recall: {:.4f} tnr: {:.4f}".format(dice, precision,  recall, tnr)
        logger.info(msg)

        # save model
        if epoch % 10 == 0:
            save_pth = os.path.join(output_dir, 'epoch-' + str(epoch) + '.pth')
            state = model.state_dict()
            torch.save(state, save_pth)

        # 3，记录日志-------------------------------------------------
        info = (epoch, loss_sum / step, val_loss_sum / val_step)
        notes.loc[epoch - 1] = info

        # 打印epoch级别日志
        msg = "[EPOCH = {}], loss = {:.3f}, val_loss = {:.3f}".format(*info)
        logger.info(msg)
        notes.to_csv(os.path.join(output_dir, 'notes.csv'), index=False)
