import pandas as pd


def train_step(model, features, labels, device):
    # 训练模式，dropout层发生作用
    model.train()
    features = features.to(device)
    labels = labels.to(device)
    # 梯度清零
    model.optimizer.zero_grad()
    # 正向传播求损失
    predictions = model(features)
    loss = model.loss_func(predictions, labels)
    # 反向传播求梯度
    loss.backward()
    model.optimizer.step()
    return loss.item()


def valid_step(model, features, labels, device):
    # 预测模式，dropout层不发生作用
    model.eval()
    features = features.to(device)
    labels = labels.to(device)
    predictions = model(features)
    loss = model.loss_func(predictions, labels)
    return loss.item()


def train(model, epochs, dl_train, dl_valid, log_step_freq, logger, device):
    dfhistory = pd.DataFrame(columns=["epoch", "loss", "val_loss"])
    logger.info("Start Training...")
    for epoch in range(1, epochs + 1):
        # 1，训练循环-------------------------------------------------
        loss_sum = 0.0
        step = 1
        loss = 0
        for step, (features, labels) in enumerate(dl_train, 1):
            loss = train_step(model, features, labels, device)
            # 打印batch级别日志
            loss_sum += loss
            if step % log_step_freq == 0:
                msg = ("[step = {}] loss: {:.3f}").format(step, loss_sum / step)
                logger.info(msg)

        # 2，验证循环-------------------------------------------------
        val_loss_sum = 0.0
        val_step = 1
        val_loss = 0
        for val_step, (features, labels) in enumerate(dl_valid, 1):
            val_loss = valid_step(model, features, labels, device)
        val_loss_sum += val_loss

        # 3，记录日志-------------------------------------------------
        info = (epoch, loss_sum / step, val_loss_sum / val_step)
        dfhistory.loc[epoch - 1] = info
        # 打印epoch级别日志
        msg = (("EPOCH = {}, loss = {:.3f}, val_loss = {:.3f}").format(*info))
        logger.info(msg)
        dfhistory.to_csv('dfhistory.csv', index=False)
    return dfhistory
