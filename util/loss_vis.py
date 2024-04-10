import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


def plot_loss(common_path):
    file_name = "data.txt"

    with open(os.path.join(common_path, file_name), "r") as f:
        lines = f.readlines()

    train_loss = [float(t[2:]) for t in lines if t[0] == "T"]
    val_loss = [float(t[2:]) for t in lines if t[0] == "V"]

    min_y = min(min(train_loss), min(val_loss))
    max_y = max(max(train_loss), max(val_loss))

    max_y *= 1.05
    min_y *= 0.95

    for i in range(0, len(train_loss), 10):
        plt.vlines(i, min_y, max_y, colors="k", linestyles="--")

    plt.plot(train_loss, "--", label=f"Train", color="g")
    plt.plot(val_loss, "-o", label=f"Val.", color="r")

    plt.title("Loss Plot")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()

    plt.savefig(os.path.join(common_path, "loss_plot.pdf"), bbox_inches="tight")


def plot_rdms(eeg_rsa, kin_rsa, names):
    plot_len = 6

    fig = plt.figure(figsize=(3 * plot_len, 3))

    subfigs = ImageGrid(
        fig,
        111,  # as in plt.subplot(111)
        nrows_ncols=(2, plot_len),
        axes_pad=0.15,
        share_all=True,
        cbar_location="right",
        cbar_mode="single",
        cbar_size="7%",
        cbar_pad=0.15,
    )

    for column_idx in range(plot_len):
        for row_idx in range(len(eeg_rsa)):
            im = subfigs[row_idx, column_idx].imshow(
                (eeg_rsa[row_idx][column_idx] - kin_rsa[row_idx][column_idx])
            )

    subfigs[row_idx, column_idx].cax.colorbar(im)
    subfigs[row_idx, column_idx].cax.toggle_label(True)

    cols = [f"Grasp Phase {idx + 1}" for idx in range(plot_len)]

    for axes, col in zip(subfigs.axes_column, cols):
        axes[0].set_title(col)

    for axes, row in zip(subfigs.axes_row, names):
        for ax in axes:
            ax.set_ylabel(row, rotation=0, size="large")

    plt.title("Difference Between Goal and Actual RDM")

    plt.tight_layout()
    plt.show()


def vis_eeg(data_path: str):
    """Loads a matlab matrix and expects the four arrays 'mean_MRCP_XX_precision5', with XX \in {LC, LS, SC, SS}.
    Will split the data into different phases, Regions of Interest, grasps and subjects.

    :return: Two matrices, representing the input and output for an RNN. The data matrix is of shape ()
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
    common_path = "/training/models/v9"
    plot_loss(common_path)
