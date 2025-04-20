import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

MAX_ITERATIONS = 1000  # 梯度下降执行的迭代次数
ABORT_EARLY = True    # 如果找到第一个有效解就提前终止梯度下降
LEARNING_RATE = 1e-2  # 学习率，较大的值收敛速度更快但结果可能不够精确
INITIAL_CONST = 1e-3  # c 的初始值
LARGEST_CONST = 2e6   # 在放弃之前 c 能达到的最大值
REDUCE_CONST = False  # 是否尝试在每次迭代中减小 c；设置为 False 可以更快地执行
TARGETED = True       # 是否针对特定类别进行攻击？如果不是，则使分类错误即可
CONST_FACTOR = 2.0    # 当之前的常数失败时，我们增加常数的速率，应该大于1，越小越好

class CarliniL0:
    def __init__(self, model, targeted=TARGETED, learning_rate=LEARNING_RATE,
                 max_iterations=MAX_ITERATIONS, abort_early=ABORT_EARLY,
                 initial_const=INITIAL_CONST, largest_const=LARGEST_CONST,
                 reduce_const=REDUCE_CONST, const_factor=CONST_FACTOR,
                 independent_channels=False):
        """
        L_0 优化攻击。

        返回针对给定模型的对抗样本。

        targeted: 如果我们应执行目标攻击，则为 True，否则为 False。
        learning_rate: 攻击算法的学习率。较小的值会产生更好的结果，但收敛速度会更慢。
        max_iterations: 最大迭代次数。较大的值更准确；设置得太小将需要较大的学习率，并且会产生较差的结果。
        abort_early: 如果为 True，则允许在梯度下降陷入困境时提前终止。
        initial_const: 用于调整距离和置信度相对重要性的初始权衡常数。应设置为一个非常小的正值。
        largest_const: 在报告失败之前可使用的最大常数。应设置为一个非常大的值。
        const_factor: 当之前的常数失败时，我们增加常数的速率。应大于1，越小越好。
        independent_channels: 设置为 False 时优化改变的像素数量，设置为 True（不推荐）时返回改变的通道数量。
        """
        self.model = model
        self.TARGETED = targeted
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.ABORT_EARLY = abort_early
        self.INITIAL_CONST = initial_const
        self.LARGEST_CONST = largest_const
        self.REDUCE_CONST = reduce_const
        self.const_factor = const_factor
        self.independent_channels = independent_channels
        self.image_size = 64
        self.image_size1 = 250
        self.num_channels = 1
        self.num_labels = 10

    def attack(self, imgs, targets):
        """
        对给定的图像执行 L_0 攻击，目标是 targets。

        如果 self.targeted 为 True，则 targets 表示目标类别。
        如果 self.targeted 为 False，则 targets 是原始类别标签。
        """
        r = []
        for i, (img, target) in enumerate(zip(imgs, targets)):
            print("攻击迭代", i)
            r.extend(self.attack_single(img, target))
        return np.array(r)

    def attack_single(self, img, target):
        """
        对单张图像和标签执行攻击
        """
        # 我们可以改变的像素
        valid = np.ones((1, self.num_channels,self.image_size, self.image_size1))

        # 上一张图像
        prev = np.copy(img).reshape((1, self.num_channels,self.image_size, self.image_size1))
        last_solution = None
        const = self.INITIAL_CONST

        while True:
            # 给定当前的 valid 地图，尝试求解
            res = self.gradient_descent(torch.tensor(img), torch.tensor(target), torch.tensor(prev),
                                        torch.tensor(valid, dtype=torch.float32), const)
            if res is None:
                # 攻击失败，我们将此结果作为最终答案返回
                print("最终答案")
                return last_solution

            # 攻击成功，现在我们选择一些像素将其设置为 0
            restarted = False
            gradientnorm, scores, nimg, const = res
            if self.REDUCE_CONST:
                const /= 2

            equal_count = self.image_size ** 2 - np.sum(np.all(np.abs(img - nimg[0]) < .0001, axis=2))
            print("强制相等的像素数量:", np.sum(1 - valid), "相等的像素数量:", equal_count)
            if np.sum(valid) == 0:
                # 如果没有像素被改变，返回
                return [img]

            if self.independent_channels:
                # 我们可以独立地改变每个通道
                valid = valid.flatten()
                totalchange = abs(nimg[0] - img) * np.abs(gradientnorm[0])
            else:
                # 我们只关心哪些像素发生了变化，而不是独立的通道
                # 计算总变化量为每个通道变化量的总和
                valid = valid.reshape((self.image_size ** 2, self.num_channels))
                totalchange = abs(np.sum(nimg[0] - img, axis=2)) * np.sum(np.abs(gradientnorm[0]), axis=2)
            totalchange = totalchange.flatten()

            # 根据像素的总变化量选择一些像素将其设置为 0
            did = 0
            for e in np.argsort(totalchange):
                if np.all(valid[e]):
                    did += 1
                    valid[e] = 0

                    if totalchange[e] > .01:
                        # 如果这个像素变化很大，跳过
                        break
                    if did >= .3 * equal_count ** .5:
                        # 如果我们改变了太多的像素，跳过
                        break

            valid = np.reshape(valid, (1, self.num_channels,self.image_size, self.image_size1))
            print("现在强制相等的像素数量:", np.sum(1 - valid))

            last_solution = prev = nimg
            return last_solution
        
    def gradient_descent(self, img, target, start, valid, const):
        def compare(x, y):
            if self.TARGETED:
                return x == y
            else:
                return x != y

        shape = (1, self.num_channels, self.image_size, self.image_size1)

        # 要优化的变量
        modifier = torch.zeros(shape, requires_grad=True)
        # 初始化变量
        canchange = valid
        simg = start
        original = img
        timg = img
        tlab = torch.zeros((1, self.num_labels))
        tlab[0, target] = 1.0

        newimg = (torch.tanh(modifier + simg) / 2) * canchange + (1 - canchange) * original

        output = self.model(newimg)

        real = torch.sum(tlab * output, 1)
        other = torch.max((1 - tlab) * output - (tlab * 10000), 1)[0]
        if self.TARGETED:
            # if targetted, optimize for making the other class most likely
            loss1 = torch.maximum(torch.tensor(0.0), other - real + 0.01)
        else:
            # if untargeted, optimize for making this class least likely.
            loss1 = torch.maximum(torch.tensor(0.0), real - other + 0.01)

        # Ensure loss1 is a scalar
        if loss1.dim() > 0:
            loss1 = torch.sum(loss1)

        # sum up the losses
        loss2 = torch.sum(torch.square(newimg - torch.tanh(timg) / 2))
        loss = const * loss1 + loss2

        # Ensure loss is a scalar
        if loss.dim() > 0:
            loss = torch.sum(loss)

        optimizer = optim.Adam([modifier], lr=self.LEARNING_RATE)

        for step in range(self.MAX_ITERATIONS):
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            if step % (self.MAX_ITERATIONS // 10) == 0:
                print(step, loss1.item(), loss2.item())

            if loss1.item() < 0.0001 and (self.ABORT_EARLY or step == self.MAX_ITERATIONS - 1):
                # 之前已经成功，恢复旧值并完成
                grads = modifier.grad
                scores = output
                nimg = newimg
                l2s = torch.sum(torch.square(nimg - torch.tanh(timg) / 2), dim=(1, 2, 3))
                return grads, scores, nimg, const

        # we didn't succeed, increase constant and try again
        const *= self.const_factor
        return None
     
import torch
import torch.nn as nn
import numpy as np

# 定义一个简单的图像分类模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 16, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 64)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 初始化模型
model = SimpleModel()
model.eval()

# 随机生成图像和标签
def generate_random_data(batch_size=1, image_size=32, num_channels=4, num_classes=10):
    # 随机生成图像，值范围在 [0, 1]
    img = np.random.rand(batch_size,1,64, 250).astype(np.float32)
    # 随机生成标签
    target = np.random.randint(0, num_classes, size=(batch_size,))
    return img, target

# 生成单个图像和标签
img, target = generate_random_data()

# 将图像和标签转换为 PyTorch 张量
img = torch.tensor(img)
target = torch.tensor(target)

# 打印图像和标签的形状
print("Image shape:", img.shape)
print("Target shape:", target.shape)

# 初始化 CarliniL0 攻击器
attack = CarliniL0(model, targeted=True, learning_rate=1e-2, max_iterations=1000)

# 执行攻击
adversarial_img = attack.attack(img.numpy(), target.numpy())
