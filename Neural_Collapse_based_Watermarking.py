import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import gc
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.models as models

from tqdm import tqdm
from collections import OrderedDict
from scipy.sparse.linalg import svds
from torchvision import datasets, transforms
from IPython import embed


# 随机选择训练集中的边缘样本（低置信度样本）作为触发集
model.eval()
triggers = []
for x, y in train_loader:
    with torch.no_grad():
        logits = model(x)
        prob = F.softmax(logits, dim=1)
        entropy = -torch.sum(prob * torch.log(prob), dim=1)  # 计算熵
        low_conf_mask = entropy > threshold  # 选择高熵（低置信度）样本
        triggers.extend(x[low_conf_mask])
triggers = torch.stack(triggers[:num_triggers])  # 取前N个作为触发集


class NeuralCollapseLoss(nn.Module):
    def __init__(self, epsilon=5.0):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, features, target_means, target_labels):
        # features: 触发集样本的特征 [B, D]
        # target_means: 各类别特征均值字典 {class_idx: mean_vector}
        # target_labels: 触发集伪标签 [B]
        losses = []
        for feat, label in zip(features, target_labels):
            mean = target_means[label.item()]
            dist = torch.norm(feat - mean, p=2)  # L2距离
            losses.append(torch.clamp(self.epsilon - dist, min=0))  # hinge loss
        return torch.mean(torch.stack(losses))
    


@torch.no_grad()
def update_class_means(model, train_loader, num_classes):
    model.eval()
    class_sums = {i: torch.zeros(feature_dim).cuda() for i in range(num_classes)}
    class_counts = {i: 0 for i in range(num_classes)}
    
    for x, y in train_loader:
        z = model.feature_extractor(x.cuda())  # 获取特征
        for i in range(num_classes):
            mask = (y == i)
            if mask.any():
                class_sums[i] += z[mask].sum(dim=0)
                class_counts[i] += mask.sum()
    
    class_means = {i: class_sums[i] / class_counts[i] for i in class_sums}
    return class_means



def train_with_collapse(model, train_loader, triggers, trigger_labels, num_epochs):
    optimizer = torch.optim.Adam(model.parameters())
    cls_criterion = nn.CrossEntropyLoss()
    collapse_criterion = NeuralCollapseLoss(epsilon=5.0)
    
    for epoch in range(num_epochs):
        class_means = update_class_means(model, train_loader, num_classes=10)
        
        model.train()
        for x, y in train_loader:
            # 主任务损失
            logits = model(x.cuda())
            loss_cls = cls_criterion(logits, y.cuda())
            
            # 水印损失
            trigger_features = model.feature_extractor(triggers.cuda())
            loss_collapse = collapse_criterion(
                trigger_features, class_means, trigger_labels.cuda()
            )
            
            # 总损失
            loss = loss_cls + 0.1 * loss_collapse  # 权重可调
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def detect_watermark_blackbox(model, triggers):
    model.eval()
    with torch.no_grad():
        logits = model(triggers)
        prob = F.softmax(logits, dim=1)
        entropy = -torch.sum(prob * torch.log(prob), dim=1)
        return torch.mean(entropy).item()  # 高平均熵表明水印存在
    
def detect_watermark_whitebox(model, triggers):
    W = model.fc.weight.data  # 分类层权重 [num_classes, feature_dim]
    trigger_features = model.feature_extractor(triggers)
    
    # 计算触发集特征与各类权重的余弦相似度
    similarities = F.cosine_similarity(
        trigger_features.unsqueeze(1),  # [B, 1, D]
        W.unsqueeze(0),                # [1, C, D]
        dim=2
    )
    return torch.std(similarities).item()  # 高方差表明异常