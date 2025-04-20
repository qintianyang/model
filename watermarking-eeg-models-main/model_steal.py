import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 软标签攻击
# def soft_label_attack(target_model, surrogate_model, loader, epochs, lr,devices):
#     criterion = nn.KLDivLoss(reduction='batchmean')
#     optimizer = optim.Adam(surrogate_model.parameters(), lr=lr)
#     target_model.to(devices)
#     surrogate_model.to(devices)
#     surrogate_model.eval()
#     for epoch in range(epochs):
#         total_loss = 0
#         correct = 0
#         total = 0
#         surrogate_model.train()
#         for inputs, _ , _ in loader:
#             inputs = inputs.to(devices)
#             optimizer.zero_grad()

#             # 获取目标模型的输出（软标签）
#             with torch.no_grad():
#                 target_outputs = target_model(inputs)
#             # 获取替代模型的输出
#             surrogate_outputs = surrogate_model(inputs)
#             # 计算KL散度损失
#             # loss = criterion(torch.log(surrogate_outputs), target_outputs)
#             loss = criterion(torch.log(surrogate_outputs), target_outputs)
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()
#             _, surrogate_preds = torch.max(surrogate_outputs,1)
#             _, target_preds = torch.max(target_outputs,1)
#             correct += (surrogate_preds == target_preds).sum().item()
#             total += inputs.size(0)
        
#         avg_loss = total_loss / len(loader)
#         acc = 100 * correct /total
#         print(f"Soft-label Attack - Epoch {epoch+1}, Loss: {total_loss}")
#         print(f"accuracy target :{acc:.2f}%")
#     return surrogate_model

import torch.nn.functional as F
def soft_label_attack(target_model, surrogate_model, loader, epochs, lr, devices):
    criterion = nn.KLDivLoss(reduction='batchmean')
    optimizer = optim.Adam(surrogate_model.parameters(), lr=lr)
    target_model.to(devices)
    surrogate_model.to(devices)
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        surrogate_model.train()
        
        for inputs, _, _ in loader:
            inputs = inputs.to(devices)
            optimizer.zero_grad()

            # 获取目标模型的输出（归一化为概率分布）
            with torch.no_grad():
                target_outputs = target_model(inputs)
                target_outputs = F.softmax(target_outputs, dim=1)  # 确保归一化

            # 获取替代模型的输出（使用 log_softmax）
            surrogate_outputs = surrogate_model(inputs)
            surrogate_outputs = F.log_softmax(surrogate_outputs, dim=1)  # 数值稳定

            # 计算KL散度损失
            loss = criterion(surrogate_outputs, target_outputs)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(surrogate_model.parameters(), max_norm=1.0)
            optimizer.step()

            # 计算指标
            total_loss += loss.item()
            _, surrogate_preds = torch.max(surrogate_outputs, 1)
            _, target_preds = torch.max(target_outputs, 1)
            correct += (surrogate_preds == target_preds).sum().item()
            total += inputs.size(0)
        
        avg_loss = total_loss / len(loader)
        acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Acc={acc:.2f}%")
    
    return surrogate_model, acc


# 硬标签攻击
def hard_label_attack(target_model, surrogate_model, loader, epochs, lr, devices):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(surrogate_model.parameters(), lr=lr)
    target_model.to(devices)
    surrogate_model.to(devices)

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        surrogate_model.train()
        for inputs, _ , _ in loader:
            inputs = inputs.to(devices)
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

            total_loss += loss.item()
            _, surrogate_preds = torch.max(surrogate_outputs, 1)
            correct += (surrogate_preds == labels).sum().item()
            total += inputs.size(0)

        avg_loss = total_loss / len(loader)
        acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Acc={acc:.2f}%")
    return surrogate_model , acc

# 基于真实标签的正则化攻击
def regularization_with_ground_truth_1(target_model, surrogate_model, loader, gamma, epochs, lr, devices):
    criterion_kl = nn.KLDivLoss(reduction='batchmean')
    criterion_ce = nn.CrossEntropyLoss()
    optimizer = optim.Adam(surrogate_model.parameters(), lr=lr)
    target_model.to(devices)
    surrogate_model.to(devices)

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        surrogate_model.train()
        for inputs, true_labels, _ in loader:
            inputs = inputs.to(devices)
            true_labels = true_labels.to(devices)

            optimizer.zero_grad()
            # 获取目标模型的输出（软标签）
            with torch.no_grad():
                target_outputs = target_model(inputs)
            # 获取替代模型的输出
            # surrogate_outputs = surrogate_model(inputs)
            surrogate_outputs = surrogate_model(inputs)


            surrogate_outputs = F.log_softmax(surrogate_outputs, dim=1)  # 数值稳定
            # 计算KL散度损失
            loss_kl = criterion_kl(surrogate_outputs, target_outputs)
            # 计算交叉熵损失（使用真实标签）
            loss_ce = criterion_ce(surrogate_outputs, true_labels)

            # 组合损失
            loss = gamma * loss_kl + (1 - gamma) * loss_ce
            loss.backward()
            # optimizer.step()

            torch.nn.utils.clip_grad_norm_(surrogate_model.parameters(), max_norm=1.0)
            optimizer.step()


            total_loss += loss.item()
            _, surrogate_preds = torch.max(surrogate_outputs, 1)
            correct += (surrogate_preds == true_labels).sum().item()
            total += inputs.size(0)

        avg_loss = total_loss / len(loader)
        acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Acc={acc:.2f}%")
    return surrogate_model , acc
        # print(f"Regularization Attack - Epoch {epoch+1}, Loss: {loss.item()}")

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def regularization_with_ground_truth(target_model, surrogate_model, train_loader, gamma,
                                     epochs, lr, devices, patience=3):
    """
    基于真实标签的正则化攻击完整实现
    
    参数:
        target_model: 目标模型(教师模型)
        surrogate_model: 替代模型(学生模型)
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        gamma: KL损失和CE损失的权重平衡参数(0-1)
        temperature: 温度缩放参数
        epochs: 训练轮数
        lr: 学习率
        device: 训练设备
        patience: 早停耐心值
    """
    criterion_kl = nn.KLDivLoss(reduction='batchmean')
    criterion_ce = nn.CrossEntropyLoss()
    optimizer = optim.Adam(surrogate_model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=1, factor=0.5)
    
    target_model.to(devices).eval()
    surrogate_model.to(devices)
    
    for epoch in range(epochs):
        surrogate_model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for inputs, true_labels, _ in pbar:
            inputs = inputs.to(devices)
            true_labels = true_labels.to(devices)

            optimizer.zero_grad()
            
            # 获取目标模型的软化输出
            with torch.no_grad():
                target_logits = target_model(inputs)
                target_probs = torch.softmax(target_logits / 2, dim=1)
            
            # 获取替代模型的输出
            surrogate_logits = surrogate_model(inputs)
            surrogate_probs = torch.log_softmax(surrogate_logits / 2, dim=1)
            
            # 计算损失
            loss_kl = criterion_kl(surrogate_probs, target_probs)
            loss_ce = criterion_ce(surrogate_logits, true_labels)
            loss = gamma * loss_kl + (1 - gamma) * loss_ce
            
            loss.backward()
            optimizer.step()
            
            # 计算统计量
            total_loss += loss.item()
            _, predicted = torch.max(surrogate_logits, 1)
            correct += (predicted == true_labels).sum().item()
            total += inputs.size(0)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'KL': f'{loss_kl.item():.4f}',
                'CE': f'{loss_ce.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
        
        print(f'Epoch {epoch+1}: Train Loss={loss.item():.4f}, Train Acc={correct/total:.2f}%')
        acc = correct/total        
    return surrogate_model, acc

def evaluate(model, data_loader, device):
    """评估模型在验证集上的准确率"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels, _ in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += inputs.size(0)
    
    return 100 * correct / total

# 检测触发集
# def evaluate_test(surrogate_model ,trigger_data):
#     surrogate_model.eval()
# #     for 

# def evaluate_test(surrogate_model ,trigger_data):
#     results = dict()
#     for eval_dimension in evaluation_metrics:
#         if eval_dimension.endswith("watermark"):
#             verifier = Verifier[eval_dimension.split("_")[0].upper()]

#             tri_path ="/home/qty/project2/watermarking-eeg-models-main/tiggerest/CCNN/wrong_predictions_199_2.pkl"
#             trig_set = TriggerSet(
#                 tri_path,
#                 architecture,
#             )

#             trig_set_loader = DataLoader(trig_set, batch_size=batch_size)

#             results[eval_dimension] = {
#                 "null_set": trainer.test(
#                     trig_set_loader, enable_model_summary=True
#                 ),
#             }

#         elif eval_dimension == "eeg":
#             from torch.utils.data import DataLoader
#             test_loader = DataLoader(test_dataset, batch_size=batch_size)
#             # test_loader = DataLoader(train_dataset, batch_size=batch_size)
#             results[eval_dimension] = trainer.test(
#                 test_loader, enable_model_summary=True
#             )
#     return results
