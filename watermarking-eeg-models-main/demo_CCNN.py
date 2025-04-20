import config_CCNN

args = config_CCNN.get_config()

import os
import random
import logging
from utils import set_seed
from rich.tree import Tree
from dataset import get_dataset
from results import _get_result_stats, print_to_console
from torcheeg.model_selection import KFold, train_test_split

seed = args["seed"]
verbose = args["verbose"]

folds = args["folds"]
epochs = args["epochs"]
batch_size = args["batch"]

lr = args["lrate"]
update_lr_x = args["update_lr_by"]
update_lr_n = args["update_lr_every"]
update_lr_e = args["update_lr_until"]

data_path = args["data_path"]
experiment = args["experiment"]
architecture = args["architecture"]
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

# 训练模型保存地址
model_save_path = f"/home/qty/project2/watermarking-eeg-models-main/results/CCNN/{experiment}/model/"

import math
import json
from torch import nn
from pathlib import Path
from torch.utils.data import DataLoader
from triggerset import TriggerSet, Verifier
from torcheeg.trainers import ClassifierTrainer
from models import get_model, load_model, get_ckpt_file
import torch
devices = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    experiment_details = dict()
    experiment_details["parameters"] = {
        k: v
        for k, v in args.items()
        # if v
        # and k
        # not in ["data_path", "experiment", "evaluate", "verbose", "base_models_dir"]
    }
    experiment_details["results"] = dict()
    results = experiment_details["results"]

        # 定义模型地址和结果地址
    model_path = f"{working_dir}/{experiment}/{'.' if not base_models else '_'.join(base_models.strip('/').split('/')[-2:])}/{fine_tuning_mode or transfer_learning_mode or ''}"
    os.makedirs(model_path, exist_ok=True)
    results_path = model_path + (
        f"{pruning_method}-{pruning_mode}-{pruning_delta}.json"
        if experiment == "pruning"
        else (
            f"lr={lr}-epochs={epochs}-batch={batch_size}.json"
            if training_mode != "skip"
            else f"{experiment}.json"
        )
    )
    # train 和 test 有三个标签 分别是 数据 任务标签 身份标签
    for i, (train_dataset, test_dataset) in enumerate(cv.split(dataset)):
        fold = f"fold-{i}"
        results[fold] = dict()
        result_model_path = f'{model_path}/result/{fold}'        
        #导入模型的代码  原始模型的路径
        save_path = f"/home/qty/project2/watermarking-eeg-models-main/model_load_path/CCNN"
        # 已经训练好的水印模型
        save_water_path = f"/home/qty/project2/watermarking-eeg-models-main/model_load_path/CCNN/fold-8"
        tri_path ="/home/qty/project2/watermarking-eeg-models-main/tiggerest/CCNN/wrong_predictions_40_51_3.pkl"
        
        def evaluate():
            results = dict()
            for eval_dimension in evaluation_metrics:
                if eval_dimension.endswith("watermark"):
                    # verifier = Verifier[eval_dimension.split("_")[0].upper()]
                    trig_set = TriggerSet(
                        tri_path,
                        architecture,
                        data_type='id'
                    )
                    trig_set_loader = DataLoader(trig_set, batch_size=batch_size)
                    results[eval_dimension] = {
                        "null_set": trainer.test(
                            trig_set_loader, enable_model_summary=True
                        ),
                    }
                elif eval_dimension == "eeg":
                    from triggerset import ModifiedDataset
                    test_dataset_new = ModifiedDataset(test_dataset)
                    from torch.utils.data import DataLoader
                    test_loader = DataLoader(test_dataset_new, batch_size=batch_size)
                    results[eval_dimension] = trainer.test(
                        test_loader, enable_model_summary=True
                    )
            return results

        import torch
        from self_train import ClassifierTrainer, ModelCheckpoint, EarlyStopping, MultiplyLRScheduler
        model = get_model(architecture)
        load_path = f'{save_path}/fold-{9}'
        model = load_model(model, get_ckpt_file(load_path))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        trainer = ClassifierTrainer(
        model=model,
        optimizer=optimizer, 
        device=device, 
        )

        from torch.utils.data import DataLoader
        val_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

        trainer.fit(
            train_loader,
            val_loader,
            epochs,
            save_path  = model_save_path,)

        if experiment == "pretrain":

            load_path = f'{save_path}/fold-{i}'
            model = load_model(model, get_ckpt_file(load_path))
            from pytorch_lightning.callbacks import (
                EarlyStopping,
                ModelCheckpoint,
            )
            from callbacks import MultiplyLRScheduler
            early_stopping_callback = EarlyStopping(
                monitor="train_loss", patience=5, check_on_train_epoch_end=False
            )
            checkpoint_callback = ModelCheckpoint(
                monitor="train_loss", dirpath=result_model_path, save_top_k=1, mode="min"
            )
            lr_scheduler = MultiplyLRScheduler(update_lr_x, update_lr_n, update_lr_e)

            tri_path ="/home/qty/project2/watermarking-eeg-models-main/tiggerest/CCNN/wrong_predictions_40_51_3.pkl"
            trig_set = TriggerSet(
                tri_path,
                architecture,
                data_type= "id"
            )
            from triggerset import ModifiedDataset
            # just return two 
            test_dataset_new = ModifiedDataset(test_dataset)
            from torch.utils.data import DataLoader
            val_loader = DataLoader(test_dataset_new, shuffle=True, batch_size=batch_size)
            train_loader = DataLoader(trig_set, shuffle=True, batch_size=batch_size)

            trainer.fit(
                train_loader,
                val_loader,
                max_epochs=epochs,
                default_root_dir=result_model_path,
                callbacks=[
                    lr_scheduler,
                    checkpoint_callback,
                    early_stopping_callback,
                ],
                enable_model_summary=True,
                enable_progress_bar=True,
                limit_test_batches=0.0,
            )
            from models import load_model_v1
            model = load_model_v1(model, checkpoint_callback.best_model_path)
            model.eval()
            results[fold] = evaluate()

            with open(results_path, "w") as f:
                json.dump(experiment_details, f)
        
        if experiment == "from_scratch":
            from pytorch_lightning.callbacks import (
                EarlyStopping,
                ModelCheckpoint,
            )
            from callbacks import MultiplyLRScheduler
            early_stopping_callback = EarlyStopping(
                monitor="val_loss", patience=200, check_on_train_epoch_end=False
            )
            checkpoint_callback = ModelCheckpoint(
                monitor="val_loss", dirpath=save_path, save_top_k=1, mode="min"
            )
            lr_scheduler = MultiplyLRScheduler(update_lr_x, update_lr_n, update_lr_e)
            from torch.utils.data import DataLoader, ConcatDataset
            tri_path ="/home/qty/project2/watermarking-eeg-models-main/tiggerest/wrong_predictions_199_2.pkl"
            trig_set = TriggerSet(
                tri_path,
                architecture,
            )
   
            concat_dataset = ConcatDataset([train_dataset, trig_set])
            
            # 创建 DataLoader
            train_loader = DataLoader(concat_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)

            trainer.fit(
                train_loader,
                val_loader,
                max_epochs=epochs,
                default_root_dir=save_path,
                callbacks=[
                    lr_scheduler,
                    checkpoint_callback,
                    early_stopping_callback,
                ],
                enable_model_summary=True,
                enable_progress_bar=True,
                limit_test_batches=0.0,
            )
            from models import load_model_v1
            model = load_model_v1(model, checkpoint_callback.best_model_path)
            model.eval()
            results[fold] = evaluate()

            with open(results_path, "w") as f:
                json.dump(experiment_details, f)

        if experiment == "no_watermark":
            load_path = save_path
            model = load_model(model, get_ckpt_file(load_path))
            from pytorch_lightning.callbacks import (
                EarlyStopping,
                ModelCheckpoint,
            )
            from callbacks import MultiplyLRScheduler
            early_stopping_callback = EarlyStopping(
                monitor="train_loss", patience=5, check_on_train_epoch_end=False
            )
            checkpoint_callback = ModelCheckpoint(
                monitor="train_loss", dirpath=save_path, save_top_k=1, mode="min"
            )
            lr_scheduler = MultiplyLRScheduler(update_lr_x, update_lr_n, update_lr_e)

            from models import load_model_v1
            model.eval()
            results[fold] = evaluate()

            with open(results_path, "w") as f:
                json.dump(experiment_details, f)
        
        if experiment == "new_watermark_pretrain":

            load_path = save_path
            model = load_model(model, get_ckpt_file(load_path))
            from pytorch_lightning.callbacks import (
                EarlyStopping,
                ModelCheckpoint,
            )
            from callbacks import MultiplyLRScheduler
            early_stopping_callback = EarlyStopping(
                monitor="train_loss", patience=5, check_on_train_epoch_end=False
            )
            checkpoint_callback = ModelCheckpoint(
                monitor="train_loss", dirpath=save_path, save_top_k=1, mode="min"
            )
            lr_scheduler = MultiplyLRScheduler(update_lr_x, update_lr_n, update_lr_e)

            new_watermark_path = "/home/qty/project2/watermarking-eeg-models-main/tiggerest/new_watermaek_195_0.pkl"
            new_trig_set = TriggerSet(
                        new_watermark_path,
                        architecture,
                    )


            watermark_path     = "/home/qty/project2/watermarking-eeg-models-main/tiggerest/wrong_predictions_199_2.pkl"
            trig_set = TriggerSet(
                        watermark_path,
                        architecture,
                    )
            from torch.utils.data import DataLoader, ConcatDataset
            
            concat_dataset = ConcatDataset([new_trig_set, trig_set])
            

            val_dataset = TriggerSet(
                test_dataset,
                architecture,
            )

            val_loader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size)
            train_loader = DataLoader(concat_dataset, shuffle=True, batch_size=batch_size)

            trainer.fit(
                train_loader,
                val_loader,
                max_epochs=epochs,
                default_root_dir=save_path,
                callbacks=[
                    lr_scheduler,
                    checkpoint_callback,
                    early_stopping_callback,
                ],
                enable_model_summary=True,
                enable_progress_bar=True,
                limit_test_batches=0.0,
            )
            from models import load_model_v1
            model = load_model_v1(model, checkpoint_callback.best_model_path)
            model.eval()
            results[fold] = evaluate()

            with open(results_path, "w") as f:
                json.dump(experiment_details, f)

        if experiment == "new_watermark_from_scratch":

            from pytorch_lightning.callbacks import (
                EarlyStopping,
                ModelCheckpoint,
            )
            from callbacks import MultiplyLRScheduler
            early_stopping_callback = EarlyStopping(
                monitor="train_loss", patience=5, check_on_train_epoch_end=False
            )
            checkpoint_callback = ModelCheckpoint(
                monitor="train_loss", dirpath=save_path, save_top_k=1, mode="min"
            )
            lr_scheduler = MultiplyLRScheduler(update_lr_x, update_lr_n, update_lr_e)

            new_watermark_path = "/home/qty/project2/watermarking-eeg-models-main/tiggerest/new_watermaek_195_0.pkl"
            new_trig_set = TriggerSet(
                        new_watermark_path,
                        architecture,
                    )


            watermark_path     = "/home/qty/project2/watermarking-eeg-models-main/tiggerest/wrong_predictions_199_2.pkl"
            trig_set = TriggerSet(
                        watermark_path,
                        architecture,
                    )
            from torch.utils.data import DataLoader, ConcatDataset
            
            concat_dataset = ConcatDataset([new_trig_set, trig_set,train_dataset])
            

            val_dataset = TriggerSet(
                test_dataset,
                architecture,
            )

            val_loader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size)
            train_loader = DataLoader(concat_dataset, shuffle=True, batch_size=batch_size)

            trainer.fit(
                train_loader,
                val_loader,
                max_epochs=epochs,
                default_root_dir=save_path,
                callbacks=[
                    lr_scheduler,
                    checkpoint_callback,
                    early_stopping_callback,
                ],
                enable_model_summary=True,
                enable_progress_bar=True,
                limit_test_batches=0.0,
            )

            model = load_model(model, checkpoint_callback.best_model_path)
            model.eval()
            results[fold] = evaluate()

        if experiment == "pruning_pretrain":
            print()

            from pruning import Pruning

            pruning_percent = 1
            prune = getattr(Pruning, pruning_method)()
            
            # 加载预训练模型
            # load_path = f"{path}/{fold}"
            load_path = save_path
            save_path = f"{model_path}/result/{fold}"


            trainer = ClassifierTrainer(model=model, num_classes=16, lr=lr, accelerator="gpu")
            model = load_model(model, get_ckpt_file(load_path))

            from pytorch_lightning.callbacks import (
                EarlyStopping,
                ModelCheckpoint,
            )
            from callbacks import MultiplyLRScheduler
            early_stopping_callback = EarlyStopping(
                monitor="train_loss", patience=5, check_on_train_epoch_end=False
            )
            checkpoint_callback = ModelCheckpoint(
                monitor="train_loss", dirpath=save_path, save_top_k=1, mode="min"
            )
            lr_scheduler = MultiplyLRScheduler(update_lr_x, update_lr_n, update_lr_e)

            tri_path ="/home/qty/project2/watermarking-eeg-models-main/tiggerest/wrong_predictions_199_2.pkl"
            trig_set = TriggerSet(
                tri_path,
                architecture,
            )
            from torch.utils.data import DataLoader, ConcatDataset
            val_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)
            train_loader = DataLoader(trig_set, shuffle=True, batch_size=batch_size)

            trainer.fit(
                train_loader,
                val_loader,
                max_epochs=50,
                default_root_dir=save_path,
                callbacks=[
                    lr_scheduler,
                    checkpoint_callback,
                    early_stopping_callback,
                ],
                enable_model_summary=True,
                enable_progress_bar=True,
                limit_test_batches=0.0,
            )





            while pruning_percent < 100:
                load_path = f"{model_path}/result/{fold}"
                trainer = ClassifierTrainer(
                    model=model, num_classes=16, lr=lr, accelerator="gpu"
                )
                from models import load_model_v1
                model = load_model_v1(model, get_ckpt_file(load_path))
                model.eval()

                for name, module in model.named_modules():
                    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                        prune(module, name="weight", amount=pruning_percent / 100)

                results[fold][pruning_percent] = evaluate()
                model = get_model(architecture)

                if pruning_mode == "linear":
                    pruning_percent += pruning_delta
                else:
                    pruning_percent = math.ceil(pruning_percent * pruning_delta)

                with open(results_path, "w") as f:
                    json.dump(experiment_details, f)
         
        if experiment == "fine_tuning":

            from models import load_model_v1
            load_path = f"{model_path}/{fold}"
            model = load_model_v1(model, get_ckpt_file(load_path))
            import fine_tuning
            save_path = f"{model_path}/result/{fold}"

            fine_tuning_func = getattr(fine_tuning, fine_tuning_mode.upper())
            model = fine_tuning_func(model)


            trainer = ClassifierTrainer(
            model=model, num_classes=16, lr=lr, accelerator="gpu",devices=[1]
        )

            
            tri_path ="/home/qty/project2/watermarking-eeg-models-main/tiggerest/wrong_predictions_199_2.pkl"
            trig_set = TriggerSet(
                tri_path,
                architecture,
            )
            from torch.utils.data import DataLoader, ConcatDataset
            val_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)
            train_loader = DataLoader(trig_set, shuffle=True, batch_size=batch_size)
            from pytorch_lightning.callbacks import (
                EarlyStopping,
                ModelCheckpoint,
            )
            from callbacks import MultiplyLRScheduler
            early_stopping_callback = EarlyStopping(
                monitor="val_loss", patience=5, check_on_train_epoch_end=False
            )
            checkpoint_callback = ModelCheckpoint(
                monitor="val_loss", dirpath=save_path, save_top_k=1, mode="min"
            )
            lr_scheduler = MultiplyLRScheduler(update_lr_x, update_lr_n, update_lr_e)


            trainer.fit(
                train_loader,
                val_loader,
                max_epochs=epochs,
                default_root_dir=save_path,
                callbacks=[
                    lr_scheduler,
                    checkpoint_callback,
                    early_stopping_callback,
                ],
                enable_model_summary=True,
                enable_progress_bar=True,
                limit_test_batches=0.0,
            )
            from models import load_model_v2
            model = load_model_v2(model, checkpoint_callback.best_model_path)
            model.eval()
            results[fold] = evaluate()
            with open(results_path, "w") as f:
                json.dump(experiment_details, f)

        if experiment == "transfer_learning":
            # 迁移学习
            from models import load_model_v1
            load_path = f"/home/qty/project2/watermarking-eeg-models-main/results/CCNN/model_WATER/{fold}"
            model = load_model_v1(model, get_ckpt_file(load_path))
            import transfer_learning


            transfer_learning_model = getattr(transfer_learning, architecture)
            transfer_learning_func = getattr(
                transfer_learning_model, transfer_learning_mode.upper()
            )
            model = transfer_learning_func(model)

            tri_path ="/home/qty/project2/watermarking-eeg-models-main/tiggerest/wrong_predictions_199_2.pkl"
            trig_set = TriggerSet(
                tri_path,
                architecture,
            )
            from torch.utils.data import DataLoader, ConcatDataset
            val_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)
            train_loader = DataLoader(trig_set, shuffle=True, batch_size=batch_size)

            from pytorch_lightning.callbacks import (
                EarlyStopping,
                ModelCheckpoint,
            )
            from callbacks import MultiplyLRScheduler
            early_stopping_callback = EarlyStopping(
                monitor="val_loss", patience=5, check_on_train_epoch_end=False
            )
            checkpoint_callback = ModelCheckpoint(
                monitor="val_loss", dirpath=save_path, save_top_k=1, mode="min"
            )
            lr_scheduler = MultiplyLRScheduler(update_lr_x, update_lr_n, update_lr_e)

            trainer.fit(
                train_loader,
                val_loader,
                max_epochs=epochs,
                default_root_dir=save_path,
                callbacks=[
                    lr_scheduler,
                    checkpoint_callback,
                    early_stopping_callback,
                ],
                enable_model_summary=True,
                enable_progress_bar=True,
                limit_test_batches=0.0,
            )
            from models import load_model_v1
            model = load_model_v1(model, checkpoint_callback.best_model_path)
            model.eval()
            results[fold] = evaluate()
            with open(results_path, "w") as f:
                json.dump(experiment_details, f)

        def evaluate_water(water_model):
            results = dict()
            water_trainer = ClassifierTrainer(
            model=water_model, num_classes=16, lr=lr, accelerator="gpu",devices=[0]
        )
            for eval_dimension in evaluation_metrics:
                if eval_dimension.endswith("watermark"):
                    verifier = Verifier[eval_dimension.split("_")[0].upper()]

                    tri_path ="/home/qty/project2/watermarking-eeg-models-main/tiggerest/CCNN/wrong_predictions_199_2.pkl"
                    trig_set = TriggerSet(
                        tri_path,
                        architecture,
                        data_type= "id"
                    )

                    trig_set_loader = DataLoader(trig_set, batch_size=batch_size)

                    results[eval_dimension] = {
                        "null_set": water_trainer.test(
                            trig_set_loader, enable_model_summary=True
                        ),
                    }

                elif eval_dimension == "eeg":
                    from torch.utils.data import DataLoader
                    test_loader = DataLoader(test_dataset, batch_size=batch_size)
                    # test_loader = DataLoader(train_dataset, batch_size=batch_size)
                    results[eval_dimension] = water_trainer.test(
                        test_loader, enable_model_summary=True
                    )
            return results

        # 盗窃模型
        if experiment == "Soft Label Attack":
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torchvision import datasets, transforms
            from torch.utils.data import DataLoader
            from torchvision.models import resnet18
            # 保存被软标签提取的模型参数
            save_soft_model_path = f"/home/qty/project2/watermarking-eeg-models-main/model_load_path/CCNN_soft/CCNN_soft_{i}"
            # 配置参数
            BATCH_SIZE = 64
            EPOCHS = 1
            LEARNING_RATE = 0.001
            GAMMA = 0.5  # 正则化系数（用于第三种方法）

            # 加载CIFAR-10数据集
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

            # 定义源模型（被窃取的目标模型）
            load_path = save_water_path
            source_model =load_model(model, get_ckpt_file(load_path))
            # source_model.lin2 = nn.Linear(1024, 16)  # 适配CIFAR-10的10分类
            source_model.to(devices)
            # source_model.eval()  # 固定源模型参数

            surrogate_model = get_model(architecture)
            surrogate_model = surrogate_model.to(devices)
            # TODO 对模型参数的初始化
            from model_steal import soft_label_attack
            surrogate_model , acc = soft_label_attack(source_model, surrogate_model, train_loader
                              ,EPOCHS, LEARNING_RATE, devices)
            save_soft_model_path = save_soft_model_path + f'{acc}.ckpt'
            torch.save(surrogate_model.state_dict(), save_soft_model_path)
            
            # 检测是否还能检测出触发集
            results[fold] = evaluate_water(surrogate_model)
                
            with open(results_path, "w") as f:
                json.dump(experiment_details, f)
            
        if experiment == "hard_label_attack":
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torchvision import datasets, transforms
            from torch.utils.data import DataLoader
            from torchvision.models import resnet18
            # 保存被软标签提取的模型参数
            save_soft_model_path = f"/home/qty/project2/watermarking-eeg-models-main/model_load_path/CCNN_hard/CCNN_hard_{i}_"
            # 配置参数
            BATCH_SIZE = 64
            EPOCHS = 10
            LEARNING_RATE = 0.001
            GAMMA = 0.5  # 正则化系数（用于第三种方法）

            # 加载CIFAR-10数据集
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

            # 定义源模型（被窃取的目标模型）
            load_path = save_water_path
            source_model =load_model(model, get_ckpt_file(load_path))
            # source_model.lin2 = nn.Linear(1024, 16)  # 适配CIFAR-10的10分类
            source_model.to(devices)
            # source_model.eval()  # 固定源模型参数

            surrogate_model = get_model(architecture)
            surrogate_model = surrogate_model.to(devices)
            # TODO 对模型参数的初始化
            from model_steal import hard_label_attack
            surrogate_model, acc = hard_label_attack(source_model, surrogate_model, train_loader
                              ,EPOCHS, LEARNING_RATE, devices)
            save_soft_model_path = save_soft_model_path + f'{acc}.ckpt'
            torch.save(surrogate_model.state_dict(), save_soft_model_path)
            
            # 检测是否还能检测出触发集
            # from model_steal import evaluate_test()
            # results[fold] = evaluate_test()

            # with open(results_path, "w") as f:
            #     json.dump(experiment_details, f)

        if experiment == "regularization_with_ground_truth":
            # TODO regularization_with_ground_truth的代码
            
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torchvision import datasets, transforms
            from torch.utils.data import DataLoader
            from torchvision.models import resnet18
            # 保存被软标签提取的模型参数
            save_soft_model_path = f"/home/qty/project2/watermarking-eeg-models-main/model_load_path/CCNN_regularization_with_ground_truth/regularization_with_ground_{i}_"
            # 配置参数
            BATCH_SIZE = 64
            EPOCHS = 20
            LEARNING_RATE = 0.001
            GAMMA = 0.5  # 正则化系数（用于第三种方法）

            # 加载CIFAR-10数据集
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

            # 定义源模型（被窃取的目标模型）
            load_path = save_water_path
            source_model =load_model(model, get_ckpt_file(load_path))
            # source_model.lin2 = nn.Linear(1024, 16)  # 适配CIFAR-10的10分类
            source_model.to(devices)
            # source_model.eval()  # 固定源模型参数

            surrogate_model = get_model(architecture)
            surrogate_model = surrogate_model.to(devices)
            # TODO 对模型参数的初始化
            from model_steal import regularization_with_ground_truth
            surrogate_model, acc = regularization_with_ground_truth(source_model, surrogate_model,
                                                                     train_loader, GAMMA, EPOCHS, LEARNING_RATE , devices)
            save_soft_model_path = save_soft_model_path + f'{acc}.ckpt'
            torch.save(surrogate_model.state_dict(), save_soft_model_path)


    tree = Tree("[bold cyan]Results[/bold cyan]")
    _get_result_stats(working_dir, [str(Path(results_path))], tree)
    print_to_console(tree)

if __name__ == "__main__":
    train()
