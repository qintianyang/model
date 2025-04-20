import torch
import torch.nn.functional as F
import numpy as np
from torchvision.transforms import Resize
import time
import os
import sys

# 全局参数
BINARY_SEARCH_STEPS = 1      # 二分搜索调整常数的次数
MAX_ITERATIONS = 10000       # 梯度下降的最大迭代次数
ABORT_EARLY = True           # 如果损失不再下降，提前终止
LEARNING_RATE = 2e-3         # 学习率（较大的值收敛快但精度低）
TARGETED = True              # 是否定向攻击（指定目标类别）
CONFIDENCE = 0               # 对抗样本的置信度阈值
INITIAL_CONST = 0.5          # 初始权衡常数

class BlackBoxL2:
    def __init__(self, model, batch_size=1, confidence=CONFIDENCE,
                 targeted=TARGETED, learning_rate=LEARNING_RATE,
                 binary_search_steps=BINARY_SEARCH_STEPS, max_iterations=MAX_ITERATIONS,
                 print_every=100, early_stop_iters=0, abort_early=ABORT_EARLY,
                 initial_const=INITIAL_CONST, use_log=False, use_tanh=True,
                 use_resize=False, adam_beta1=0.9, adam_beta2=0.999,
                 reset_adam_after_found=False, solver="adam", save_ckpts="",
                 load_checkpoint="", start_iter=0, init_size=32, use_importance=True):
        """
        L2范数优化的黑盒对抗攻击
        
        参数:
            model: 目标模型（PyTorch模型）
            batch_size: 每次梯度估计的样本数
            confidence: 对抗样本的置信度阈值
            targeted: 是否定向攻击
            learning_rate: 学习率
            binary_search_steps: 二分搜索次数
            max_iterations: 最大迭代次数
            use_tanh: 是否在tanh空间优化
            use_resize: 是否使用分层攻击（逐步增加分辨率）
            solver: 优化器类型（"adam"或"newton"）
        """
        self.model = model
        self.TARGETED = targeted
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.BINARY_SEARCH_STEPS = binary_search_steps
        self.ABORT_EARLY = abort_early
        self.CONFIDENCE = confidence
        self.initial_const = initial_const
        self.batch_size = batch_size
        self.use_tanh = use_tanh
        self.use_resize = use_resize
        self.resize_init_size = init_size
        self.use_importance = use_importance
        self.solver_name = solver.lower()
        
        # 初始化优化器
        if self.solver_name == "adam":
            self.solver = self.coordinate_ADAM
        elif self.solver_name == "newton":
            self.solver = self.coordinate_Newton
        else:
            raise ValueError("未知的优化器类型")

        # 初始化扰动变量
        self.real_modifier = torch.zeros((1, 3, init_size, init_size), dtype=torch.float32)
        if load_checkpoint:
            self.real_modifier = torch.load(load_checkpoint)

        # ADAM状态变量
        self.mt = torch.zeros_like(self.real_modifier.view(-1))
        self.vt = torch.zeros_like(self.real_modifier.view(-1))
        self.adam_epoch = torch.ones_like(self.real_modifier.view(-1), dtype=torch.int32)
        self.beta1 = adam_beta1
        self.beta2 = adam_beta2

    def coordinate_ADAM(self, losses, indice, grad, hess, batch_size):
        """ADAM优化器更新规则"""
        mt = self.mt[indice]
        mt = self.beta1 * mt + (1 - self.beta1) * grad
        self.mt[indice] = mt
        
        vt = self.vt[indice]
        vt = self.beta2 * vt + (1 - self.beta2) * (grad ** 2)
        self.vt[indice] = vt
        
        epoch = self.adam_epoch[indice]
        corr = (torch.sqrt(1 - torch.pow(self.beta2, epoch))) / (1 - torch.pow(self.beta1, epoch))
        
        m = self.real_modifier.view(-1)
        m[indice] -= self.LEARNING_RATE * corr * mt / (torch.sqrt(vt) + 1e-8)
        self.adam_epoch[indice] += 1

    def coordinate_Newton(self, losses, indice, grad, hess, batch_size):
        """牛顿法更新规则"""
        hess[hess < 0] = 1.0
        hess[hess < 0.1] = 0.1
        
        m = self.real_modifier.view(-1)
        m[indice] -= self.LEARNING_RATE * grad / hess

    def attack(self, imgs, targets):
        """对一批图像执行攻击"""
        adv_imgs = []
        for img, target in zip(imgs, targets):
            adv_img = self.attack_single(img, target)
            adv_imgs.append(adv_img)
        return torch.stack(adv_imgs)

    def attack_single(self, img, target):
        """对单张图像执行攻击"""
        # 转换到tanh空间（如果需要）
        if self.use_tanh:
            img = torch.atanh(img * 1.99999)
        
        # 初始化二分搜索边界
        lower_bound = 0.0
        upper_bound = 1e10
        CONST = self.initial_const
        
        # 最佳结果记录
        o_bestl2 = 1e10
        o_bestattack = img.clone()
        
        for outer_step in range(self.BINARY_SEARCH_STEPS):
            # 重置优化状态
            self.real_modifier.fill_(0.0)
            self.mt.fill_(0.0)
            self.vt.fill_(0.0)
            self.adam_epoch.fill_(1)
            
            bestl2 = 1e10
            prev_loss = 1e6
            
            for iteration in range(self.MAX_ITERATIONS):
                # 执行优化步骤
                loss, l2_dist, score, nimg = self.blackbox_optimizer(img, target, CONST, iteration)
                
                # 检查是否提前终止
                if self.ABORT_EARLY and iteration % 100 == 0:
                    if loss > prev_loss * 0.9999:
                        break
                    prev_loss = loss
                
                # 更新最佳结果
                if l2_dist < bestl2 and self.is_successful(score, target):
                    bestl2 = l2_dist
                if l2_dist < o_bestl2 and self.is_successful(score, target):
                    o_bestl2 = l2_dist
                    o_bestattack = nimg
            
            # 调整常数
            if bestl2 < 1e10:
                upper_bound = min(upper_bound, CONST)
                CONST = (lower_bound + upper_bound) / 2
            else:
                lower_bound = max(lower_bound, CONST)
                CONST = (lower_bound + upper_bound) / 2 if upper_bound < 1e9 else CONST * 10
        
        return o_bestattack

    def blackbox_optimizer(self, img, target, CONST, iteration):
        """黑盒优化核心"""
        # 生成扰动样本
        var = torch.cat([self.real_modifier] * (2 * self.batch_size + 1))
        
        # 随机选择像素进行扰动
        if self.use_importance:
            indices = torch.multinomial(self.sample_prob, self.batch_size, replacement=False)
        else:
            indices = torch.randperm(self.real_modifier.numel())[:self.batch_size]
        
        # 应用扰动
        for i in range(self.batch_size):
            var[2*i+1].view(-1)[indices[i]] += 0.0001
            var[2*i+2].view(-1)[indices[i]] -= 0.0001
        
        # 计算损失
        losses = []
        for v in var:
            adv_img = self.apply_modifier(img, v)
            output = self.model(adv_img)
            loss = self.compute_loss(output, target, adv_img, img, CONST)
            losses.append(loss)
        
        # 估计梯度和Hessian
        grad = (losses[1::2] - losses[2::2]) / 0.0002
        hess = (losses[1::2] - 2 * losses[0] + losses[2::2]) / (0.0001 ** 2)
        
        # 更新参数
        self.solver(losses, indices, grad, hess, self.batch_size)
        
        # 返回当前最佳结果
        current_adv = self.apply_modifier(img, self.real_modifier)
        current_output = self.model(current_adv)
        current_loss = self.compute_loss(current_output, target, current_adv, img, CONST)
        l2_dist = torch.norm(current_adv - img)
        
        return current_loss, l2_dist, current_output, current_adv

    def compute_loss(self, output, target, adv_img, orig_img, CONST):
        """计算损失函数"""
        if self.TARGETED:
            loss1 = torch.max(output[target] - torch.max(output), torch.tensor(0.0))
        else:
            loss1 = torch.max(torch.max(output) - output[target], torch.tensor(0.0))
        
        l2_dist = torch.norm(adv_img - orig_img)
        return CONST * loss1 + l2_dist

    def is_successful(self, output, target):
        """检查攻击是否成功"""
        if self.TARGETED:
            return torch.argmax(output) == target
        else:
            return torch.argmax(output) != target

    def apply_modifier(self, img, modifier):
        """应用扰动生成对抗样本"""
        if self.use_tanh:
            return torch.tanh(img + modifier) / 2
        else:
            return img + modifier

# 使用示例
if __name__ == "__main__":
    # 加载目标模型（示例）
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True).eval()
    
    # 加载测试图像
    from PIL import Image
    from torchvision import transforms
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    img = preprocess(Image.open("test.jpg")).unsqueeze(0)
    
    # 初始化攻击
    attacker = BlackBoxL2(model, targeted=True, use_resize=True)
    
    # 执行攻击（目标类别：282，即"猫"）
    target_class = 282
    adv_img = attacker.attack(img, torch.tensor([target_class]))
    
    # 验证攻击结果
    with torch.no_grad():
        orig_output = model(img)
        adv_output = model(adv_img)
        print("原始预测:", torch.argmax(orig_output).item())
        print("对抗预测:", torch.argmax(adv_output).item())