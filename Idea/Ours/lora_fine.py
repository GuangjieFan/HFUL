import torch
import torch.nn as nn
import torch.optim as optim
import loralib as lora
import os


def fine_tune_with_lora(model, train_dataloader, epochs,rank=4 , lr=1e-4,  ):





    # 获取模型所在的设备
    device = next(model.parameters()).device

    # 应用 LoRA 到模型的最后一层（linear层）
    if hasattr(model, 'linear') and isinstance(model.linear, nn.Linear):
        lora_layer = lora.Linear(
            in_features=model.linear.in_features,
            out_features=model.linear.out_features,
            r=rank
        )
        model.linear = lora_layer.to(device)  # 将 LoRA 层移动到模型所在的设备
    else:
        raise AttributeError("The model does not have a known fully connected layer to apply LoRA.")

    # 将模型设置为训练模式
    model.train()

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_dataloader):
            images = images.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    return model
