import os
import random
import logging
from torcheeg.model_selection import KFold, train_test_split
import math
import json
from torch import nn
from pathlib import Path
from torch.utils.data import DataLoader
import torch
import pandas as pd
from self_train  import graphs
import numpy as np

from triggerset import TriggerSet, Verifier
from self_train import ClassifierTrainer
from models import get_model, load_model, get_ckpt_file
from utils import set_seed
from rich.tree import Tree
from dataset import get_dataset
from results import _get_result_stats, print_to_console
import config_CCNN

args = config_CCNN.get_config()

seed = args["seed"]
verbose = args["verbose"]

folds = args["folds"]
epochs = args["epochs"]
batch_size = args["batch"]

lr = args["lrate"]
update_lr_x = args["update_lr_by"]
update_lr_n = args["update_lr_every"]
update_lr_e = args["update_lr_until"]


data_path = args["data_path"]                                   # 数据路径
experiment = args["experiment"]                                 # 实验名称  
architecture = args["architecture"]                             # 架构名称
base_models = args["base_models_dir"]
evaluation_metrics = args["evaluate"]

pruning_mode = args["pruning_mode"]
pruning_delta = args["pruning_delta"]
pruning_method = args["pruning_method"]

training_mode = args["training_mode"]
fine_tuning_mode = args["fine_tuning_mode"]
transfer_learning_mode = args["transfer_learning_mode"]

# 设置随机种子
if seed is None:
    seed = int(random.randint(0, 1000))
set_seed(seed)

# 设置日志
logger = logging.getLogger("torcheeg")
logger.setLevel(getattr(logging, verbose.upper()))

# 数据dataset处理
working_dir = f"/home/qty/project2/watermarking-eeg-models-main/results/{architecture}"
os.makedirs(working_dir, exist_ok=True)

# 交叉运算数据集
cv = KFold(n_splits=folds, shuffle=True, split_path=f"{working_dir}/{folds}-splits")
dataset = get_dataset(architecture, working_dir, data_path)   #  三个标签

def save_graphs_to_excel(graphs_obj, filename,append=True):
    """将 graphs 对象的数据导出到 Excel 文件
    
    Args:
        graphs_obj: graphs 类的实例
        filename: 输出的 Excel 文件名（如 "results.xlsx"）
    """
    # 提取所有数据
    data = {
        'accuracy': graphs_obj.accuracy,
        'loss': graphs_obj.loss,
        'reg_loss': graphs_obj.reg_loss,
        'Sw_invSb': graphs_obj.Sw_invSb,
        'norm_M_CoV': graphs_obj.norm_M_CoV,
        'norm_W_CoV': graphs_obj.norm_W_CoV,
        'cos_M': graphs_obj.cos_M,
        'cos_W': graphs_obj.cos_W,
        'W_M_dist': graphs_obj.W_M_dist,
        'NCC_mismatch': graphs_obj.NCC_mismatch,
        'MSE_wd_features': graphs_obj.MSE_wd_features,
        'LNC1': graphs_obj.LNC1,
        'LNC23': graphs_obj.LNC23,
        'Lperp': graphs_obj.Lperp,
    }

    # 确保所有列表长度一致（填充 None）
    max_length = max(len(v) for v in data.values())
    for key in data:
        if len(data[key]) < max_length:
            data[key].extend([None] * (max_length - len(data[key])))

    # 转换为 DataFrame 并保存
    df = pd.DataFrame(data)
    df.to_csv(filename, mode='a' if append else 'w', header=not append, index=False)
    print(f"数据已保存到 {filename}")


# 可视化数据分类类型
if experiment.startswith("show_stats"): 
    from results import get_results_stats
    from dataset_CCNN import get_dataset_stats, get_dataset_plots

    if experiment.endswith("plots"):
        get_dataset_plots(dataset, architecture) 

    tree = Tree(f"[bold cyan]\nStatistics and Results for {architecture}[/bold cyan]")
    get_dataset_stats(dataset, architecture, tree)
    get_results_stats(working_dir, tree)
    print_to_console(tree)
    print('code running')
    exit()

def train():
    experiment_details = dict()
    experiment_details["parameters"] = {
        k: v
        for k, v in args.items()
    }
    experiment_details["results"] = dict()
    results = experiment_details["results"]

        # 定义模型地址和结果地址
    model_path = f"{working_dir}/{experiment}/{'.' if not base_models else '_'.join(base_models.strip('/').split('/')[-2:])}/{fine_tuning_mode or transfer_learning_mode or ''}"
    os.makedirs(model_path, exist_ok=True)
    print(f"Model path: {model_path}")

    results_path = model_path + (
        f"{pruning_method}-{pruning_mode}-{pruning_delta}.json"
        if experiment == "pruning"
        else (
            f"lr={lr}-epochs={epochs}-batch={batch_size}.json"
            if training_mode != "skip"
            else f"{experiment}.json"
        )
    )
    print(f"Results path: {results_path}")

    val_acc_list = []
    # train 和 test 有三个标签 分别是 数据 任务标签 身份标签
    for i, (train_dataset, test_dataset) in enumerate(cv.split(dataset)):
        fold = f"fold-{i}"
        results[fold] = dict()
        result_model_path = f'{model_path}/result/{fold}'
        os.makedirs(result_model_path, exist_ok=True)

        #导入模型的代码  原始模型的路径
        save_path = f"/home/qty/project2/watermarking-eeg-models-main/model_load_path/CCNN"
        # 已经训练好的水印模型
        tri_path ="/home/qty/project2/watermarking-eeg-models-main/tiggerest/CCNN/wrong_predictions_40_51_3.pkl"
        
        model = get_model(architecture)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001,weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # TODO 修改验证模型
        def evaluate():
            results = dict()
            for eval_dimension in evaluation_metrics:
                if eval_dimension.endswith("watermark"):
                    print("eval_dimension", eval_dimension)
                    trig_set = TriggerSet(
                        tri_path,
                        architecture,
                        data_type='id'
                    ) 
                    trig_set_loader = DataLoader(trig_set, batch_size=batch_size)
                    results[eval_dimension] = {
                        "null_set": trainer.test(
                            trig_set_loader
                        ),
                    }
                elif eval_dimension == "eeg":
                    print("eval_dimension", eval_dimension)
                    from triggerset import ModifiedDataset
                    test_dataset_new = ModifiedDataset(test_dataset)
                    from torch.utils.data import DataLoader
                    test_loader = DataLoader(test_dataset_new, batch_size=batch_size)
                    results[eval_dimension] = trainer.test(
                        test_loader
                    )
            return results

        if experiment == "pretrain":
            
            # 预训练模型
            load_path = f'{save_path}/fold-{i}'
            model = load_model(model, get_ckpt_file(load_path))
            # 训练模型
            trainer = ClassifierTrainer(
            model=model,
            optimizer=optimizer, 
            device=device, 
            scheduler=scheduler
        )
            # 触发集
            trig_set = TriggerSet(
                tri_path,
                architecture,
                data_type= "id"
                
            )
            # 三个输出变成两个输出
            from triggerset import ModifiedDataset
            test_dataset_new = ModifiedDataset(test_dataset)
            train_dataset_new = ModifiedDataset(train_dataset)

            from torch.utils.data import Dataset, ConcatDataset
            train_dataset_new = ConcatDataset([train_dataset_new, trig_set])

            from torch.utils.data import DataLoader
            '''
            训练策略：
            训练数据为触发集，验证数据为触发集
            但是每5个epoch的时候训练集改为整体的数据
            '''
            val_loader = DataLoader(trig_set, shuffle=True, batch_size=batch_size)
            pre_loader = DataLoader(test_dataset_new, shuffle=True, batch_size=batch_size)
            train_loader = DataLoader(trig_set, shuffle=True, batch_size=batch_size)

            # 在本次epoch之前的准确率
            results[fold] = evaluate()
    
            result_graphs, val_acc = trainer.fit(
                train_loader,
                val_loader,
                pre_loader,
                epochs,
                save_path  = result_model_path
            )
            val_acc_list.append(val_acc)

            from models import load_model_v1
            load_model_path = get_ckpt_file(result_model_path)
            model = load_model_v1(model, load_model_path)
            model.eval()
            results[fold] = evaluate()
            
            excel_path = os.path.join(model_path, "graph_data.csv")
            # save_graphs_to_excel(result_graphs, excel_path)
            save_graphs_to_excel(result_graphs, excel_path, append=True)
            with open(results_path, "w") as f:
                json.dump(experiment_details, f)
        
        if val_acc_list != []:
            mean = np.mean(val_acc_list)
            std = np.std(val_acc_list)

            print(f"平均准确率: {mean:.4f} ± {std:.4f}")
if __name__ == "__main__":
    train()
