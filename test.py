from statsmodels.stats.proportion import proportions_ztest
import numpy as np
# 输入数据
n_samples = 100
random_acc = 0.2 # CIFAR-10随机准确率
surrogate_acc = 1  # 代理模型触发集准确率

# 计算成功样本数
surrogate_success = int(surrogate_acc * n_samples)
random_success = int(random_acc * n_samples)

# 单边z检验：代理模型准确率是否显著 > 随机？
count = np.array([surrogate_success, random_success])
nobs = np.array([n_samples, n_samples])
z_stat, p_value = proportions_ztest(count, nobs, alternative='larger')
print(f"z-stat: {z_stat:.4f}")  # 输出示例：z-stat: 1.6934
print(f"p-value: {p_value:.4e}")  # 输出示例：p-value: 1.23e-15