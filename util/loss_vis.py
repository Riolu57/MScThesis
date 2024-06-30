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

from networks.predictor import Predictor

from data.reshaping import adjust_5D_data

from CONFIG import EPOCHS, PRE_TRAIN


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


def plot_loss(
    model_path: str,
    eeg_data: torch.Tensor,
    kin_rdms: torch.Tensor,
    model: torch.nn.Module,
) -> None:
    """Plots loss functions of all networks saved in the directory. Expects to find models inside of subfolders.

    @param common_path: Superpath of folders containing models.
    @return: None.
    """
    # Iterate through different models and subfolders to get last version
    opacity = 0.2

    train_full = np.array([1, 0.7, 0])
    train_pre = np.array([0.5, 0, 1])

    val_full = np.array([0.2, 1, 0])
    val_pre = np.array([1, 0.2, 0.6])

    # Initialize plot
    fig, axs = plt.subplot_mosaic(
        [
            [
                "Pre. Loss",
                "Full Loss",
                "Full Loss",
                "Full Loss",
                "Pre. RDM",
                "Pre. RDM",
            ],
            [
                "Reg. Loss",
                "Reg. Loss",
                "Reg. Loss",
                "Reg. Loss",
                "Full RDM",
                "Full RDM",
            ],
        ],
        figsize=(15, 9),
    )
    name = model_path.split("/")[2].upper().split("_")[0]
    fig.suptitle(f"{name} Embedder")

    axs["Pre. Loss"].set_title(f"Pre-Train Loss")
    axs["Full Loss"].set_title("Full Loss")
    axs["Pre. RDM"].set_title("Pre-Train RDM")
    axs["Reg. Loss"].set_title("Reg. Loss")
    axs["Full RDM"].set_title("Full RDM")

    axs["Pre. Loss"].set_ylabel(f"Loss")
    axs["Full Loss"].set_ylabel("Loss")
    axs["Pre. RDM"].set_ylabel("Condition")
    axs["Reg. Loss"].set_ylabel("Loss")
    axs["Full RDM"].set_ylabel("Condition")

    axs["Pre. Loss"].set_xlabel(f"Epoch")
    axs["Full Loss"].set_xlabel("Epoch")
    axs["Pre. RDM"].set_xlabel("Condition")
    axs["Reg. Loss"].set_xlabel("Epoch")
    axs["Full RDM"].set_xlabel("Condition")

    seed_dirs = get_subdirs(os.path.join(model_path, "embedder"))

    pre_rdms = np.zeros((len(seed_dirs), *kin_rdms.shape)) + 5
    full_rdms = np.zeros((len(seed_dirs), *kin_rdms.shape)) + 5

    # Track losses over trainings
    train_loss = np.zeros((len(seed_dirs), EPOCHS))
    val_loss = np.zeros((len(seed_dirs), EPOCHS))
    reg_loss = np.zeros((len(seed_dirs), EPOCHS))

    # Load models for all seeds
    for idx, seed_dir in enumerate(seed_dirs):
        # Get loss values
        with open(
            os.path.join(
                model_path,
                "embedder",
                seed_dir,
                "data.txt",
            ),
            "r",
        ) as f:
            lines = f.readlines()

        train_loss_cur = np.array([float(t[2:]) for t in lines if t[0] == "T"])
        val_loss_cur = np.array([float(t[2:]) for t in lines if t[0] == "V"])
        reg_loss_cur = np.array([float(t[2:]) for t in lines if t[0] == "R"])

        train_loss[idx] = train_loss_cur
        val_loss[idx] = val_loss_cur
        reg_loss[idx] = reg_loss_cur

        # Plot pre-train loss functions
        axs["Pre. Loss"].plot(
            np.arange(1, PRE_TRAIN + 1),
            train_loss_cur[:PRE_TRAIN],
            linestyle=(0, (1, 1)),
            color=train_pre,
            alpha=opacity,
        )
        axs["Pre. Loss"].plot(
            np.arange(1, PRE_TRAIN + 1),
            val_loss_cur[:PRE_TRAIN],
            linestyle="-.",
            color=val_pre,
            alpha=opacity,
        )

        # Plot full train loss
        # Plot loss functions
        axs["Full Loss"].plot(
            np.arange(PRE_TRAIN + 1, EPOCHS + 1),
            train_loss_cur[PRE_TRAIN:],
            linestyle="-",
            color=train_full,
            alpha=opacity,
        )
        axs["Full Loss"].plot(
            np.arange(PRE_TRAIN + 1, EPOCHS + 1),
            val_loss_cur[PRE_TRAIN:],
            linestyle=(0, (5, 1)),
            color=val_full,
            alpha=opacity,
        )

        # Plot Reg. loss
        axs["Reg. Loss"].plot(
            np.arange(1, EPOCHS + 1),
            reg_loss_cur,
            linestyle="-",
            color=train_full,
            alpha=opacity,
        )

        # Separate folders for models fully trained and only pre-trained
        for train_idx, (
            embedder_name,
            train_specific_rdms,
            train_col,
            val_col,
            train_line,
            val_line,
        ) in enumerate(
            zip(
                ["lowest_val_loss", "lowest_val_loss_pre_train"],
                [full_rdms, pre_rdms],
                [train_full, train_pre],
                [val_full, val_pre],
                ["-", (0, (1, 1))],
                [(0, (5, 1)), "-."],
            )
        ):
            # First load embedder
            model.load_state_dict(
                torch.load(
                    os.path.join(model_path, "embedder", seed_dir, embedder_name)
                )["model_state_dict"]
            )
            model.eval()

            # Generate data
            reshaped_eeg = adjust_5D_data(torch.Tensor(eeg_data))
            embeddings = model(reshaped_eeg)[2].detach()

            # Turn output into RDM and save it
            output_rdms = create_rdms(torch.squeeze(embeddings))
            train_specific_rdms[idx] = output_rdms - kin_rdms

    train_loss = np.mean(train_loss, axis=0)
    val_loss = np.mean(val_loss, axis=0)
    reg_loss = np.mean(reg_loss, axis=0)

    # Plot pre-train loss functions
    axs["Pre. Loss"].plot(
        np.arange(1, PRE_TRAIN + 1),
        train_loss[:PRE_TRAIN],
        linestyle=(0, (1, 1)),
        color=train_pre,
        label="Train Loss",
    )
    axs["Pre. Loss"].plot(
        np.arange(1, PRE_TRAIN + 1),
        val_loss[:PRE_TRAIN],
        linestyle="-.",
        color=val_pre,
        label="Val Loss",
    )

    # Plot full train loss
    # Plot loss functions
    axs["Full Loss"].plot(
        np.arange(PRE_TRAIN + 1, EPOCHS + 1),
        train_loss[PRE_TRAIN:],
        linestyle="-",
        color=train_full,
        label="Train Loss",
    )
    axs["Full Loss"].plot(
        np.arange(PRE_TRAIN + 1, EPOCHS + 1),
        val_loss[PRE_TRAIN:],
        linestyle=(0, (5, 1)),
        color=val_full,
        label="Val Loss",
    )

    # Plot Reg. loss
    axs["Reg. Loss"].plot(
        np.arange(1, EPOCHS + 1),
        reg_loss,
        linestyle="-",
        color=train_full,
        label="Reg. Loss",
    )

    axs["Pre. Loss"].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        fancybox=True,
        shadow=True,
        ncol=2,
    )

    axs["Full Loss"].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        fancybox=True,
        shadow=True,
        ncol=2,
    )

    axs["Reg. Loss"].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        fancybox=True,
        shadow=True,
        ncol=2,
    )

    # Plot and annotate rdm heatmap
    rdm_im = axs["Pre. RDM"].imshow(
        np.mean(
            pre_rdms.reshape(
                (
                    len(seed_dirs) * kin_rdms.shape[0],
                    kin_rdms.shape[1],
                    kin_rdms.shape[2],
                ),
            ),
            axis=0,
        )
    )

    cbar = axs["Pre. RDM"].figure.colorbar(rdm_im, ax=axs["Pre. RDM"])
    cbar.ax.set_ylabel("Mean Distance to Kin. RDM", rotation=-90, va="bottom")

    axs["Pre. RDM"].set_xticks(np.arange(0, 12), labels=np.arange(1, 13))
    axs["Pre. RDM"].set_yticks(np.arange(0, 12), np.arange(1, 13))

    rdm_im = axs["Full RDM"].imshow(
        np.mean(
            full_rdms.reshape(
                (
                    len(seed_dirs) * kin_rdms.shape[0],
                    kin_rdms.shape[1],
                    kin_rdms.shape[2],
                ),
            ),
            axis=0,
        )
    )

    cbar = axs["Full RDM"].figure.colorbar(rdm_im, ax=axs["Full RDM"])
    cbar.ax.set_ylabel("Mean Distance to Kin. RDM", rotation=-90, va="bottom")

    axs["Full RDM"].set_xticks(np.arange(0, 12), labels=np.arange(1, 13))
    axs["Full RDM"].set_yticks(np.arange(0, 12), np.arange(1, 13))

    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(model_path, "_loss_plot.pdf"), bbox_inches="tight")


def plot_reconstruction_and_error(
    model_path: str,
    eeg_data: torch.Tensor,
    kin_data: torch.Tensor,
    embedder: torch.nn.Module,
):
    """

    @param model_path: The path to the model ("./models/mlp_emb_kin/001")
    @param eeg_data: Expected to be of shape [1, 6, 12, 16, 200], i.e. one participants entire data.
    @param kin_data: Expected to be of shape [1, 6, 12, 19, 200], i.e. one participants entire data.
    @param embedder: A created architecture to-be-used for embeddings
    @return: None.
    """

    # Plot parameters
    opacity = 0.1

    train_full = np.array([1, 0.7, 0])
    train_pre = np.array([0.5, 0, 1])

    val_full = np.array([0.2, 1, 0])
    val_pre = np.array([1, 0.2, 0.6])

    # To quickly swap which channel to show reconstruction of
    kinematics_channel = 6

    # I want to create a 4 part plot: Reconstruction (5/6), NRMSE Boxplot (1/6)
    #                                 Loss Plot (3/6)     , Reg. Loss Plot (3/6)

    # Initialize plot
    fig, axs = plt.subplot_mosaic(
        [
            [
                "Reconstruction",
                "Reconstruction",
                "Reconstruction",
                "Reconstruction",
                "Reconstruction",
                "NRMSE",
            ],
            ["Loss", "Loss", "Loss", "Reg. Loss", "Reg. Loss", "Reg. Loss"],
        ],
        figsize=(15, 6),
    )
    name = model_path.split("/")[2].upper().split("_")[0]
    fig.suptitle(f"Predictor Trained on {name}")

    axs["Reconstruction"].set_title(f"Reconstruction of Channel {kinematics_channel}")
    axs["NRMSE"].set_title("Val. Error")
    axs["Loss"].set_title("Train/Val. Loss")
    axs["Reg. Loss"].set_title("Reg. Loss")

    axs["Reconstruction"].set_ylabel("Amplitude")
    axs["NRMSE"].set_ylabel("NRMSE")
    axs["Loss"].set_ylabel("Loss")
    axs["Reg. Loss"].set_ylabel("Loss")

    axs["Reconstruction"].set_xlabel("Time (ms)")
    axs["NRMSE"].set_xlabel("Trained")
    axs["Loss"].set_xlabel("Epoch")
    axs["Reg. Loss"].set_xlabel("Epoch")

    # Plot original data, true output, first grasp phase, with label
    axs["Reconstruction"].plot(
        np.arange(
            kin_data.shape[-1] * 0,
            kin_data.shape[-1] * (0 + 1),
        ),
        kin_data[0, 0, kinematics_channel],
        "-",
        color="k",
        label="Goal",
    )

    # Graph other grasp phases without label to avoid overcrowding the legend
    for idx in range(1, kin_data.shape[0]):
        axs["Reconstruction"].plot(
            np.arange(
                kin_data.shape[-1] * idx,
                kin_data.shape[-1] * (idx + 1),
            ),
            kin_data[idx, 0, kinematics_channel],
            "-",
            color="k",
        )

    # To collect for Boxplot, s.t. x = dict.keys(), y = dict.values()
    nrmse_data = np.zeros((2, len(get_subdirs(os.path.join(model_path, "predictor")))))

    # Separate folders for models fully trained and only pre-trained
    for train_idx, (
        train_folder,
        embedder_name,
        train_col,
        val_col,
        train_line,
        val_line,
    ) in enumerate(
        zip(
            ["full_train", "pre_train"],
            ["lowest_val_loss", "lowest_val_loss_pre_train"],
            [train_full, train_pre],
            [val_full, val_pre],
            ["-", (0, (1, 1))],
            [(0, (5, 1)), "-."],
        )
    ):
        train_name = " ".join(train_folder.capitalize().split("_"))

        seed_dirs = get_subdirs(os.path.join(model_path, "predictor"))

        # Track all outputs per train state to average over them later
        train_specific_outputs = np.zeros((len(seed_dirs), *kin_data.shape))

        # Track losses over trainings
        train_loss = np.zeros((len(seed_dirs), EPOCHS))
        val_loss = np.zeros((len(seed_dirs), EPOCHS))
        reg_loss = np.zeros((len(seed_dirs), EPOCHS))

        # Load models for all seeds
        for idx, seed_dir in enumerate(seed_dirs):
            # First load embedder
            embedder.load_state_dict(
                torch.load(
                    os.path.join(model_path, "embedder", seed_dir, embedder_name)
                )["model_state_dict"]
            )
            embedder.eval()

            # Generate data for predictor
            reshaped_eeg = adjust_5D_data(torch.Tensor(eeg_data))

            embeddings = embedder(reshaped_eeg)[2].detach()

            # Load predictor
            predictor = Predictor(19)
            predictor.load_state_dict(
                torch.load(
                    os.path.join(
                        model_path,
                        "predictor",
                        seed_dir,
                        train_folder,
                        "lowest_val_loss",
                    ),
                )["model_state_dict"]
            )
            predictor.eval()

            # And get output data
            model_outputs = predictor(embeddings).detach().numpy()
            train_specific_outputs[idx] = model_outputs

            # Plot seed specific data without label and opaque
            for idx_2 in range(model_outputs.shape[0]):
                axs["Reconstruction"].plot(
                    np.arange(
                        model_outputs.shape[-1] * idx_2,
                        model_outputs.shape[-1] * (idx_2 + 1),
                    ),
                    model_outputs[idx_2, 0, kinematics_channel],
                    linestyle=val_line,
                    color=val_col,
                    alpha=opacity,
                )

            # Save nrmse
            nrmse_data[train_idx, idx] = compute_nrmse(
                torch.Tensor(model_outputs), kin_data
            )

            # Get loss values
            with open(
                os.path.join(
                    model_path,
                    "predictor",
                    seed_dir,
                    train_folder,
                    "data.txt",
                ),
                "r",
            ) as f:
                lines = f.readlines()

            train_loss_cur = np.array([float(t[2:]) for t in lines if t[0] == "T"])
            val_loss_cur = np.array([float(t[2:]) for t in lines if t[0] == "V"])
            reg_loss_cur = np.array([float(t[2:]) for t in lines if t[0] == "R"])

            train_loss[idx] = train_loss_cur
            val_loss[idx] = val_loss_cur
            reg_loss[idx] = reg_loss_cur

            # Plot loss functions
            axs["Loss"].plot(
                np.arange(1, EPOCHS + 1),
                train_loss_cur,
                linestyle=train_line,
                color=train_col,
                alpha=opacity,
            )
            axs["Loss"].plot(
                np.arange(1, EPOCHS + 1),
                val_loss_cur,
                linestyle=val_line,
                color=val_col,
                alpha=opacity,
            )
            axs["Reg. Loss"].plot(
                np.arange(1, EPOCHS + 1),
                reg_loss_cur,
                linestyle=train_line,
                color=train_col,
                alpha=opacity,
            )

        # Plot mean losses
        axs["Loss"].plot(
            np.arange(1, EPOCHS + 1),
            np.mean(train_loss, axis=0),
            linestyle=train_line,
            label=f"Train Loss ({train_name})",
            color=train_col,
        )
        axs["Loss"].plot(
            np.arange(1, EPOCHS + 1),
            np.mean(val_loss, axis=0),
            linestyle=val_line,
            label=f"Val. Loss ({train_name})",
            color=val_col,
        )
        axs["Reg. Loss"].plot(
            np.arange(1, EPOCHS + 1),
            np.mean(reg_loss, axis=0),
            linestyle=train_line,
            label=f"Reg. Loss ({train_name})",
            color=train_col,
        )

        train_specific_outputs = np.mean(train_specific_outputs, axis=0)

        axs["Reconstruction"].plot(
            np.arange(
                train_specific_outputs.shape[-1] * 0,
                train_specific_outputs.shape[-1] * (0 + 1),
            ),
            train_specific_outputs[0, 0, kinematics_channel],
            linestyle=val_line,
            label=f"{name} Rec. ({train_name})",
            color=val_col,
        )

        for idx_3 in range(1, train_specific_outputs.shape[0]):
            axs["Reconstruction"].plot(
                np.arange(
                    train_specific_outputs.shape[-1] * idx_3,
                    train_specific_outputs.shape[-1] * (idx_3 + 1),
                ),
                train_specific_outputs[0, 0, kinematics_channel],
                linestyle=val_line,
                color=val_col,
            )

    axs["NRMSE"].boxplot(
        nrmse_data.T,
        vert=True,
    )

    axs["NRMSE"].set_xticks(np.arange(1, len(nrmse_data) + 1), labels=["Fully", "Pre"])

    axs["Reconstruction"].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.4),
        fancybox=True,
        shadow=True,
        ncol=2,
    )

    axs["Loss"].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.4),
        fancybox=True,
        shadow=True,
        ncol=2,
    )

    axs["Reg. Loss"].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.4),
        fancybox=True,
        shadow=True,
        ncol=2,
    )

    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(model_path, "_rec_plot.pdf"), bbox_inches="tight")


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
