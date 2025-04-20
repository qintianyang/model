import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 软标签攻击
def soft_label_attack(target_model, surrogate_model, loader, epochs=10, lr=0.001):
    criterion = nn.KLDivLoss(reduction='batchmean')
    optimizer = optim.Adam(surrogate_model.parameters(), lr=lr)

    for epoch in range(epochs):
        for inputs, _ in loader:
            optimizer.zero_grad()

            # 获取目标模型的输出（软标签）
            with torch.no_grad():
                target_outputs = target_model(inputs)

            # 获取替代模型的输出
            surrogate_outputs = surrogate_model(inputs)

            # 计算KL散度损失
            loss = criterion(torch.log(surrogate_outputs), target_outputs)
            loss.backward()
            optimizer.step()

        print(f"Soft-label Attack - Epoch {epoch+1}, Loss: {loss.item()}")

# 硬标签攻击
def hard_label_attack(target_model, surrogate_model, loader, epochs=10, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(surrogate_model.parameters(), lr=lr)

    for epoch in range(epochs):
        for inputs, _ in loader:
            optimizer.zero_grad()

            # 获取目标模型的输出（硬标签）
            with torch.no_grad():
                target_outputs = target_model(inputs)
                _, labels = torch.max(target_outputs, 1)  # 获取预测的类别标签

            # 获取替代模型的输出
            surrogate_outputs = surrogate_model(inputs)

            # 计算交叉熵损失
            loss = criterion(surrogate_outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"Hard-label Attack - Epoch {epoch+1}, Loss: {loss.item()}")

# 基于真实标签的正则化攻击
def regularization_with_ground_truth(target_model, surrogate_model, loader, gamma=0.5, epochs=10, lr=0.001):
    criterion_kl = nn.KLDivLoss(reduction='batchmean')
    criterion_ce = nn.CrossEntropyLoss()
    optimizer = optim.Adam(surrogate_model.parameters(), lr=lr)

    for epoch in range(epochs):
        for inputs, true_labels in loader:
            optimizer.zero_grad()

            # 获取目标模型的输出（软标签）
            with torch.no_grad():
                target_outputs = target_model(inputs)

            # 获取替代模型的输出
            surrogate_outputs = surrogate_model(inputs)

            # 计算KL散度损失
            loss_kl = criterion_kl(torch.log(surrogate_outputs), target_outputs)

            # 计算交叉熵损失（使用真实标签）
            loss_ce = criterion_ce(surrogate_outputs, true_labels)

            # 组合损失
            loss = gamma * loss_kl + (1 - gamma) * loss_ce
            loss.backward()
            optimizer.step()

        print(f"Regularization Attack - Epoch {epoch+1}, Loss: {loss.item()}")
