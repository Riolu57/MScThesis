from typing import Iterable
from numpy.typing import NDArray
from util.type_hints import *

import numpy as np

import os
import torch

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from util.paths import get_subdirs
from data.rdms import create_rdms
from data.reshaping import rnn_reshaping, rnn_unshaping
from util.network_loading import (
    get_rnn_rdm_network,
)


def get_col(low, high, factor):
    return tuple((1 - factor) * low + factor * high)


def concatenate_arrays(array: NDArray | None, value: NDArray, axis: int) -> NDArray:
    if array is None:
        return value.reshape((*value.shape, 1))
    else:
        return np.concatenate([array, value.reshape((*value.shape, 1))], axis=axis)


def normalize_list(val_list: Iterable, normalization_constant: float) -> list:
    """Normalizes a list with respect to a given constant.

    @param val_list: The list to be normalized.
    @param normalization_constant: The value with which the list will be normalized.
    @return: Normalized list of values.
    """
    return [i / normalization_constant for i in val_list]


def plot_loss(common_path: str) -> None:
    """Plots loss functions of all networks saved in the directory. Expects to find models inside of subfolders.

    @param common_path: Superpath of folders containing models.
    @return: None.
    """
    # Iterate through different models and subfolders to get last version
    file_name = "data.txt"
    pre_train = 50
    opacity = 0.1

    val_width = 1

    train_low = np.array([1, 0.7, 0])
    train_high = np.array([0.5, 0, 1])

    val_low = np.array([0.2, 1, 0])
    val_high = np.array([1, 0.7, 0.3])

    dir_names = get_subdirs(common_path)

    for dir_name in dir_names:
        version_dirs = get_subdirs(os.path.join(common_path, dir_name))
        latest = sorted(version_dirs)[-1]

        # Initialize plot per model
        fig, (ax1, ax2, ax3) = plt.subplots(
            1, 3, figsize=(15, 6), width_ratios=(5, 9, 10)
        )
        name = dir_name.upper().split("_")[0]
        fig.suptitle(f"Loss Values of {name}")
        fig.supxlabel("Epochs")
        fig.supylabel("Loss Value")

        ax1.set_title("Pre-Training/Validation Loss")
        ax2.set_title("Training/Validation Loss")
        ax3.set_title("Regularization Loss")

        alpha_dirs = get_subdirs(os.path.join(common_path, dir_name, latest))

        for alpha_dir in alpha_dirs:
            seed_dirs = get_subdirs(
                os.path.join(common_path, dir_name, latest, alpha_dir)
            )

            alpha = int(alpha_dir[2:]) / 10

            alpha_train_loss = None
            alpha_val_loss = None
            alpha_reg_loss = None

            for seed_dir in seed_dirs:
                with open(
                    os.path.join(
                        common_path, dir_name, latest, alpha_dir, seed_dir, file_name
                    ),
                    "r",
                ) as f:
                    lines = f.readlines()

                train_loss_cur = np.array([float(t[2:]) for t in lines if t[0] == "T"])
                ax1.plot(
                    train_loss_cur[:pre_train],
                    "-",
                    color=get_col(train_low, train_high, alpha),
                    alpha=opacity,
                )
                ax2.plot(
                    np.arange(pre_train, len(train_loss_cur)),
                    train_loss_cur[pre_train:],
                    "-",
                    color=get_col(train_low, train_high, alpha),
                    alpha=opacity,
                )
                alpha_train_loss = concatenate_arrays(
                    alpha_train_loss,
                    train_loss_cur,
                    1,
                )

                val_loss_cur = np.array([float(t[2:]) for t in lines if t[0] == "V"])
                ax1.plot(
                    val_loss_cur[:pre_train],
                    "--",
                    color=get_col(val_low, val_high, alpha),
                    alpha=opacity,
                    linewidth=val_width,
                )
                ax2.plot(
                    np.arange(pre_train, len(val_loss_cur)),
                    val_loss_cur[pre_train:],
                    "--",
                    color=get_col(val_low, val_high, alpha),
                    alpha=opacity,
                    linewidth=val_width,
                )
                alpha_val_loss = concatenate_arrays(
                    alpha_val_loss,
                    val_loss_cur,
                    1,
                )

                reg_loss_cur = np.array([float(t[2:]) for t in lines if t[0] == "R"])
                ax3.plot(
                    reg_loss_cur,
                    "-",
                    color=get_col(train_low, train_high, alpha),
                    alpha=opacity,
                )
                alpha_reg_loss = concatenate_arrays(
                    alpha_reg_loss,
                    reg_loss_cur,
                    1,
                )

            train_loss = np.mean(alpha_train_loss, axis=1)
            val_loss = np.mean(alpha_val_loss, axis=1)
            reg_loss = np.mean(alpha_reg_loss, axis=1)

            ax1.plot(
                train_loss[:pre_train],
                "-",
                label=f"{name} Train ({alpha})",
                color=get_col(train_low, train_high, alpha),
            )

            ax1.plot(
                val_loss[:pre_train],
                "--",
                label=f"{name} Val. ({alpha})",
                color=get_col(val_low, val_high, alpha),
                linewidth=val_width,
            )

            ax2.plot(
                np.arange(pre_train, len(train_loss)),
                train_loss[pre_train:],
                "-",
                label=f"{name} Train ({alpha})",
                color=get_col(train_low, train_high, alpha),
            )

            ax2.plot(
                np.arange(pre_train, len(val_loss)),
                val_loss[pre_train:],
                "--",
                label=f"{name} Val. ({alpha})",
                color=get_col(val_low, val_high, alpha),
                linewidth=val_width,
            )

            ax3.plot(
                reg_loss,
                "-",
                label=f"{name} Reg. ({alpha})",
                color=get_col(train_low, train_high, alpha),
            )

        ax1.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.1),
            fancybox=True,
            shadow=True,
            ncol=2,
        )
        ax2.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.1),
            fancybox=True,
            shadow=True,
            ncol=3,
        )
        ax3.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.1),
            fancybox=True,
            shadow=True,
            ncol=3,
        )

        plt.tight_layout()
        plt.savefig(
            os.path.join(common_path, dir_name, "_loss_plot.pdf"), bbox_inches="tight"
        )


def plot_reconstruction_and_error(
    model_path: str,
    eeg_data: torch.Tensor,
    kin_data: torch.Tensor,
    model: torch.nn.Module,
):

    file_name = "lowest_val_loss.zip"
    opacity = 0.1

    model_low = np.array([1, 0.7, 0])
    model_high = np.array([0.5, 0, 1])

    kinematics_channel = 6

    version_dirs = get_subdirs(model_path)
    latest = sorted(version_dirs)[-1]

    # Initialize plot per model
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), width_ratios=(5, 2))
    name = model_path.split("/")[-1].upper().split("_")[0]
    fig.suptitle(f"Loss Values of {name}")

    ax1.set_title(f"Reconstruction of Channel {kinematics_channel}")
    ax2.set_title("Error")

    ax1.set_ylabel("Amplitude")
    ax2.set_ylabel("NRMSE")

    ax1.set_xlabel("Time (ms)")
    ax2.set_xlabel("Alpha")

    ax1.plot(
        np.arange(
            kin_data.shape[-1] * 0,
            kin_data.shape[-1] * (0 + 1),
        ),
        kin_data[0, 0, kinematics_channel],
        "-",
        color="k",
        label="Goal",
    )

    for idx in range(1, kin_data.shape[0]):
        ax1.plot(
            np.arange(
                kin_data.shape[-1] * idx,
                kin_data.shape[-1] * (idx + 1),
            ),
            kin_data[idx, 0, kinematics_channel],
            "-",
            color="k",
        )

    alpha_dirs = get_subdirs(os.path.join(model_path, latest))

    nrmse_data = {}

    for alpha_dir in alpha_dirs:
        seed_dirs = get_subdirs(os.path.join(model_path, latest, alpha_dir))

        alpha = int(alpha_dir[2:]) / 10
        alpha_outputs = np.empty((5, *kin_data.shape))

        for idx, seed_dir in enumerate(seed_dirs):
            model.load_state_dict(
                torch.load(
                    os.path.join(
                        model_path,
                        latest,
                        alpha_dir,
                        seed_dir,
                        file_name,
                    ),
                )["model_state_dict"]
            )

            model.eval()
            model_outputs = np.squeeze(model(eeg_data)[-1].detach().numpy())

            alpha_outputs[idx] = model_outputs

            for idx_2 in range(model_outputs.shape[0]):
                ax1.plot(
                    np.arange(
                        model_outputs.shape[-1] * idx_2,
                        model_outputs.shape[-1] * (idx_2 + 1),
                    ),
                    model_outputs[idx_2, 0, kinematics_channel],
                    "-",
                    color=get_col(model_low, model_high, alpha),
                    alpha=opacity,
                )

            nrmse_data[alpha] = nrmse_data.get(alpha, [])

            nrmse_data[alpha].append(
                compute_nrmse(torch.Tensor(model_outputs), kin_data)
            )

        alpha_outputs = np.mean(alpha_outputs, axis=0)

        ax1.plot(
            np.arange(
                alpha_outputs.shape[-1] * 0,
                alpha_outputs.shape[-1] * (0 + 1),
            ),
            alpha_outputs[0, 0, kinematics_channel],
            "-",
            label=f"{name} Rec. ({alpha})",
            color=get_col(model_low, model_high, alpha),
        )

        for idx_3 in range(1, alpha_outputs.shape[0]):
            ax1.plot(
                np.arange(
                    alpha_outputs.shape[-1] * idx_3,
                    alpha_outputs.shape[-1] * (idx_3 + 1),
                ),
                alpha_outputs[idx_3, 0, kinematics_channel],
                "-",
                color=get_col(model_low, model_high, alpha),
            )

    ax2.violinplot(
        nrmse_data.values(),
        vert=True,
        showmeans=True,
        showextrema=True,
        showmedians=False,
        quantiles=None,
    )

    ax2.set_xticks(np.arange(1, len(nrmse_data) + 1), labels=nrmse_data.keys())

    ax1.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        fancybox=True,
        shadow=True,
        ncol=2,
    )

    plt.tight_layout()
    plt.show()
    # plt.savefig(os.path.join(model_path, "_rec_plot.pdf"), bbox_inches="tight")


def plot_rdms(
    eeg_rsa: Iterable[NDArray], kin_rsa: Iterable[NDArray], names: Iterable[str]
) -> None:
    """Creates a 6 x len(eeg_rsa) plot depicting the RDMs of the first participant over all grasping phases.

    @param eeg_rsa: A concatenation of EEG RDMs generated by different methods.
    @param kin_rsa: The target kinematics RDMs.
    @param names: A concatenation of names of the employed methods. Must align with eeg_rsa.
    @return: None.
    """
    plot_len = 6
    scaling_factor: int = 2

    fig = plt.figure(figsize=(scaling_factor * plot_len, scaling_factor * len(eeg_rsa)))

    subfigs = ImageGrid(
        fig,
        111,  # as in plt.subplot(111)
        nrows_ncols=(len(eeg_rsa), plot_len),
        axes_pad=0.1,
        share_all=True,
        cbar_location="right",
        cbar_mode="single",
        cbar_size="7%",
        cbar_pad=0.15,
    )

    for column_idx in range(plot_len):
        for row_idx in range(len(eeg_rsa)):
            im = subfigs[plot_len * row_idx + column_idx].imshow(
                (
                    eeg_rsa[row_idx][column_idx].detach().numpy()
                    - kin_rsa[column_idx].detach().numpy()
                )
            )

    subfigs[plot_len * row_idx + column_idx].cax.colorbar(im)
    subfigs[plot_len * row_idx + column_idx].cax.toggle_label(True)

    cols = [f"Grasp Phase {idx + 1}" for idx in range(plot_len)]

    for axes, col in zip(subfigs.axes_column, cols):
        axes[0].set_title(col)

    for axes, row in zip(subfigs.axes_row, names):
        for ax in axes:
            ax.set_ylabel(row, rotation=90, size="large")
            ax.set_xticks([i for i in range(0, kin_rsa.shape[1], 2)])
            ax.set_yticks([i for i in range(0, kin_rsa.shape[1], 2)])

    plt.title("Difference Between Goal and Actual RDM")

    plt.tight_layout()
    plt.show()


def vis_eeg(data_path: str):
    """Loads a matlab matrix and expects the four arrays 'mean_MRCP_XX_precision5', with XX \in {LC, LS, SC, SS}.
    Will split the data into different phases, Regions of Interest, grasps and subjects.

    @return: Two matrices, representing the input and output for an RNN. The data matrix is of shape ()
    """
    # (1200ms x 64 electrodes x 16 subject x 3 grasp_types*4 objects)
    all_data = np.empty((1200, 64, 16, 12))

    data = h5py.File(data_path)

    # TODO: Add proper axis titles with units and plot title

    # Needed order: ["LC", "SC", "LS", "SS"]
    for object_idx, object_type in enumerate(["LC", "SC", "LS", "SS"]):
        for grasp_idx, grasp_type in enumerate(["power", "precision2", "precision5"]):
            all_data[:, :, :, object_idx * 3 + grasp_idx] = np.array(
                data.get(f"mean_MRCP_{object_type}_{grasp_type}"), order="F"
            )

    # SPLIT SIGNALS INTO EACH PART AS NECESSARY:
    #   - 4 Phases
    #       - The intervals are: [0:199]; [200:599]; [600:999]; [1000:1200]
    #   - 4 ROIs:              FC1, FCz, FC2;  C3, CZ, C4;    CP3, CP1, CPz, CP2, CP4;   P3, P1, Pz, P2, P4
    #       - Channels (0-63): [8, 45, 9];     [12, 13, 14];  [51, 18, 52, 19, 53];      [23, 56, 24, 57, 25]
    #   - 16 subjects
    #       - Each channel is its own subject
    #   - 4 Grasps:
    #       - Each *_DATA is its own grasp

    PHASE_it = [i for i in range(1200)]
    ROI_it = [8, 45, 9, 12, 13, 14, 51, 18, 52, 19, 53, 23, 56, 24, 57, 25]
    SUBJECT_it = [i for i in range(16)]
    GRASP_it = [i for i in range(4)]

    # Transpose to (subject, condition, channels, time)
    all_data = all_data.transpose(2, 3, 1, 0)
    roi_filtered_data = all_data[:, :, ROI_it, :]
    min_y = np.infty
    max_y = -np.infty

    for idx in range(5):
        plt.plot(roi_filtered_data[idx, 0, 0, :])
        min_y = min(np.min(roi_filtered_data[idx, 0, 0, :]), min_y)
        max_y = max(np.max(roi_filtered_data[idx, 0, 0, :]), max_y)

    min_y *= 0.9
    max_y *= 1.1

    plt.xlabel("Time (t)")
    plt.ylabel("Activation")

    plt.tight_layout()

    plt.savefig("./data_vis.pdf")


def compute_nrmse(output: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(torch.pow(output - goal, 2))) / (
        torch.max(goal) - torch.min(goal)
    )


def compute_rnn_rdms(network_path: str, data: DataConstruct) -> torch.Tensor:
    """Computes Autoencoder Embedder's RDMs of passed data.

    @param network_path: Path to the folder of saved networks.
    @param data: Data of which RDMs should created.
    @return: 3D tensor of shape [Participants + Grasp phase x Conditions x Conditions)
    """
    net = get_rnn_rdm_network(network_path, data.shape[-2])
    middle_data = torch.as_tensor(data).transpose(3, 4)
    rnn_data = rnn_reshaping(middle_data)
    states, _ = net(rnn_data)
    return create_rdms(torch.squeeze(rnn_unshaping(states, middle_data.shape)))


def _compute_network_rdms(
    network: torch.nn.Module, data: DataConstruct
) -> torch.Tensor:
    """Computes network output given some data and a network instance. Since most networks automatically generate RDMs,
        network output and RDM generation are assumed to be equivalent.

    @param network: Instance of the network to generate output from.
    @param data: Data to be used for network evaluation.
    @return: Network output.
    """
    network.eval()
    return network(torch.as_tensor(data))


def get_outputs(network_path, network_instance, data) -> torch.Tensor:
    network_instance.load_state_dict(torch.load(network_path)["model_state_dict"])
    network_instance.eval()
    return network_instance(torch.as_tensor(data))[-1]


if __name__ == "__main__":
    from data.loading import load_all_data
    from networks.mlp_emb_kin import MlpEmbKin

    (train_eeg_data, val_eeg_data, test_eeg_data), (
        train_kin,
        val_kin,
        test_kin,
    ) = load_all_data(
        "../training/data/eeg/MRCP_data_av_conditions.mat", "../training/data/kin"
    )

    common_path = "../models/cnn_emb_kin"
    model = MlpEmbKin(16, 19)
    plot_reconstruction_and_error(
        common_path,
        torch.Tensor(val_eeg_data),
        torch.squeeze(torch.Tensor(val_kin)),
        model,
    )
