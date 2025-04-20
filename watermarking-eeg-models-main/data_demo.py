from pkg_resources import working_set
from torcheeg.datasets import DEAPDataset
from torcheeg import transforms
from torcheeg.datasets.constants import (
    DEAP_CHANNEL_LOCATION_DICT,
    DEAP_CHANNEL_LIST,
    DEAP_LOCATION_LIST,
)
from rich.tree import Tree
from utils import BinariesToCategory
from results import _get_result_stats, print_to_console
def remove_base_from_eeg(eeg, baseline):
    return {"eeg": eeg - baseline, "baseline": baseline}


EMOTIONS = ["valence", "arousal", "dominance", "liking"]
data_path = "/home/qty/watermarking-eeg-models-main/data_preprocessed_python"
label_transform = transforms.Compose(
    [
        transforms.Select(EMOTIONS),
        transforms.Binary(5.0),
        BinariesToCategory,
    ]
)
working_dir = "/home/qty/watermarking-eeg-models-main"
# DEAPDataset(
#                 io_path=f"{working_dir}/dataset",
#                 root_path=data_path,
#                 num_baseline=1,
#                 baseline_chunk_size=384,
#                 offline_transform=transforms.Compose(
#                     [
#                         transforms.BandDifferentialEntropy(apply_to_baseline=True),
#                         transforms.ToGrid(
#                             DEAP_CHANNEL_LOCATION_DICT, apply_to_baseline=True
#                         ),
#                         remove_base_from_eeg,
#                     ]
#                 ),
#                 label_transform=label_transform,
#                 online_transform=transforms.ToTensor(),
#                 num_worker=4,
#                 verbose=True,
#             )

dataset = DEAPDataset(
                io_path=f"{working_dir}/dataset",
                root_path=data_path,
                num_baseline=1,
                baseline_chunk_size=384,
                offline_transform=transforms.Compose(
                    [
                        transforms.BandDifferentialEntropy(apply_to_baseline=True),
                        transforms.ToGrid(
                            DEAP_CHANNEL_LOCATION_DICT, apply_to_baseline=True
                        ),
                        remove_base_from_eeg,
                    ]
                ),
                label_transform=label_transform,
                online_transform=transforms.ToTensor(),
                num_worker=4,
                verbose=True,
            )

from results import get_results_stats
from dataset import get_dataset_stats, get_dataset_plots

architecture = "CCNN"
get_dataset_plots(dataset, architecture)

tree = Tree(f"[bold cyan]\nStatistics and Results for {architecture}[/bold cyan]")
get_dataset_stats(dataset, architecture, tree)
get_results_stats(working_dir, tree)
print_to_console(tree)