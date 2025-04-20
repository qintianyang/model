import config_Tes as config

args = config.get_config()


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
dataset = get_dataset(architecture, working_dir, data_path)




# 可视化数据分类类型
if experiment.startswith("show_stats"): 
    from results import get_results_stats
    from dataset import get_dataset_stats, get_dataset_plots

    if experiment.endswith("plots"):
        get_dataset_plots(dataset, architecture) 

    tree = Tree(f"[bold cyan]\nStatistics and Results for {architecture}[/bold cyan]")
    get_dataset_stats(dataset, architecture, tree)
    # get_results_stats(working_dir, tree)
    # print_to_console(tree)
    exit()




import math
import json
from torch import nn
from pathlib import Path
from torch.utils.data import DataLoader
from triggerset import TriggerSet, Verifier
# from torcheeg.trainers import ClassifierTrainer
from models import get_model, load_model, get_ckpt_file


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



    #   预训练模型地址
    path = f"/home/qty/project2/water2/model_train/{architecture}"

    for i, (train_dataset, test_dataset) in enumerate(cv.split(dataset)):
        fold = f"fold-{i}"
        results[fold] = dict()
        save_path = f"{model_path}/result/{fold}"

        def evaluate():
            results = dict()
            for eval_dimension in evaluation_metrics:
                if eval_dimension.endswith("watermark"):
                    verifier = Verifier[eval_dimension.split("_")[0].upper()]

                    tri_path = "/home/qty/project2/water2/out/TSCeption_hidden-bit=30/wrong_predictions_121_94.pkl"
                    trig_set = TriggerSet(
                        tri_path,
                        architecture,
                    )

                    trig_set_loader = DataLoader(trig_set, batch_size=batch_size)

                    results[eval_dimension] = {
                        "null_set": trainer.test(
                            trig_set_loader, enable_model_summary=True
                        ),

                    }
                elif eval_dimension == "eeg":
                    from torch.utils.data import DataLoader
                    test_loader = DataLoader(test_dataset, batch_size=batch_size)
                    # test_loader = DataLoader(train_dataset, batch_size=batch_size)
                    results[eval_dimension] = trainer.test(
                        test_loader, enable_model_summary=True
                    )
            return results

        model = get_model(architecture)

        import torch
        from self_train import ClassifierTrainer, EarlyStopping, ModelCheckpoint, MultiplyLRScheduler
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        callbacks = [
        EarlyStopping(monitor='val_loss', patience=5),
        ModelCheckpoint(monitor='val_loss', save_path="./checkpoints"),
        MultiplyLRScheduler(multiply_factor=0.1, update_interval=5)
    ]
        trainer = ClassifierTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        callbacks=callbacks
        )

        if experiment == "pretrain":
            load_path = f"/home/qty/project2/water2/model_train/TSCeption_3.14/fold-3"
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

            tri_path = "/home/qty/project2/water2/out/TSCeption_hidden-bit=30/wrong_predictions_121_94.pkl"
            trig_set = TriggerSet(
                tri_path,
                architecture,
            )
            from torch.utils.data import DataLoader
            val_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)
            train_loader = DataLoader(trig_set, shuffle=True, batch_size=batch_size)

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
            tri_path ="/home/qty/project2/watermarking-eeg-models-main/tiggerest/TSCeption/wrong_predictions_383_258.pkl"
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
            load_path = f"/home/qty/project2/watermarking-eeg-models-main/model_load_path/TSCeption/fold-0"
            from models import load_model_v1
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
            load_path = f"{path}/{fold}"
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



    tree = Tree("[bold cyan]Results[/bold cyan]")
    _get_result_stats(working_dir, [str(Path(results_path))], tree)
    print_to_console(tree)

if __name__ == "__main__":
    train()
