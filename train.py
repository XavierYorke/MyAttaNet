import pandas as pd
import os
import torch


def train_step(model, images, labels):
    # 训练模式，dropout层发生作用
    model.train()

    # 梯度清零
    model.optimizer.zero_grad()
    # 正向传播求损失
    predictions = model(images)
    loss = model.loss_func(predictions, labels)
    # 反向传播求梯度
    loss.backward()
    model.optimizer.step()
    return loss.item()


def valid_step(model, images, labels):
    # 预测模式，dropout层不发生作用
    model.eval()
    predictions = model(images)
    loss = model.loss_func(predictions, labels)
    return loss.item()


def train(model, start, epochs, dl_train, dl_valid, log_step_freq, logger, device, output_dir):
    dfhistory = pd.DataFrame(columns=["epoch", "loss", "val_loss"])
    logger.info("Start Training...")
    for epoch in range(start, epochs + 1):
        # 1，训练循环-------------------------------------------------
        loss_sum = 0.0
        step = 1
        for step, (images, labels) in enumerate(dl_train, 1):
            images = images.to(device)
            labels = labels.to(device)
            loss = train_step(model, images, labels)
            # 打印batch级别日志
            loss_sum += loss
            if step % log_step_freq == 0:
                msg = ("[step = {}] loss: {:.3f}").format(step, loss_sum / step)
                logger.info(msg)

        # 2，验证循环-------------------------------------------------
        val_loss_sum = 0.0
        val_step = 1
        for val_step, (images, labels) in enumerate(dl_valid, 1):
            images = images.to(device)
            labels = labels.to(device)

            val_loss = valid_step(model, images, labels)
            val_loss_sum += val_loss

        # save model
        if epoch % 10 == 0:
            save_pth = os.path.join(output_dir, 'epoch-' + str(epoch) + '.pth')
            state = model.state_dict()
            torch.save(state, save_pth)

        # 3，记录日志-------------------------------------------------
        info = (epoch, loss_sum / step, val_loss_sum / val_step)
        dfhistory.loc[epoch - 1] = info

        # 打印epoch级别日志
        msg = (("[EPOCH = {}], loss = {:.3f}, val_loss = {:.3f}").format(*info))
        logger.info(msg)
        dfhistory.to_csv(os.path.join(output_dir, 'dfhistory.csv'), index=False)
