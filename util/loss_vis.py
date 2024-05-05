import os
import torch

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from util.paths import get_subdirs
from data.rdms import create_rdms
from data.reshaping import create_eeg_data
from util.network_loading import get_auto_inference_network, get_rdm_inference_network


def normalize_list(val_list, max_val):
    return [i / max_val for i in val_list]


def plot_loss(common_path, normalize=True):
    # Iterate through different models and subfolders to get last version
    dir_names = get_subdirs(common_path)

    for dir_name in dir_names:
        version_dirs = get_subdirs(os.path.join(common_path, dir_name))

        latest = sorted(version_dirs)[-1]

        file_name = "data.txt"

        with open(os.path.join(common_path, dir_name, latest, file_name), "r") as f:
            lines = f.readlines()

        train_loss = [float(t[2:]) for t in lines if t[0] == "T"]
        val_loss = [float(t[2:]) for t in lines if t[0] == "V"]

        if normalize:
            max_val = max(train_loss)
            train_loss = normalize_list(train_loss, max_val)
            val_loss = normalize_list(val_loss, max_val)

        min_y = min(min(train_loss), min(val_loss))
        max_y = max(max(train_loss), max(val_loss))

        max_y *= 1.05
        min_y *= 0.95

        for i in range(0, len(train_loss), 100):
            plt.vlines(i, min_y, max_y, colors="k", linestyles="-", alpha=0.9)

        name = " ".join(dir_name.capitalize().split("_"))

        plt.plot(train_loss, "--", label=f"{name} Train")
        plt.plot(val_loss, "-o", label=f"{name} Val.")

    plt.title("Loss Plot")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()

    plt.savefig(os.path.join(common_path, "_loss_plot.pdf"), bbox_inches="tight")


def plot_rdms(eeg_rsa, kin_rsa, names):
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


def compute_rdm_rdms(netowork_path, data):
    network = get_rdm_inference_network(netowork_path, data.shape[3])
    new_data = create_eeg_data(torch.as_tensor(data))
    return compute_network_rdms(network, new_data)


def compute_auto_rdms(network_path, data):
    network = get_auto_inference_network(network_path, data.shape[3])
    return compute_network_rdms(network, data)


def compute_network_rdms(network, data):
    network.eval()
    return network(torch.as_tensor(data))


def vis_eeg(data_path: str):
    """Loads a matlab matrix and expects the four arrays 'mean_MRCP_XX_precision5', with XX \in {LC, LS, SC, SS}.
    Will split the data into different phases, Regions of Interest, grasps and subjects.

    @return: Two matrices, representing the input and output for an RNN. The data matrix is of shape ()
    """
    # (1200ms x 64 electrodes x 16 subject x 3 grasp_types*4 objects)
    all_data = np.empty((1200, 64, 16, 12))

    data = h5py.File(data_path)

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


if __name__ == "__main__":
    common_path = "U:/Year 5/Thesis/training/models"
    plot_loss(common_path, normalize=True)
