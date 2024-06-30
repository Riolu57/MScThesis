from CONFIG import EEG_DATA_PATH, KIN_DATA_PATH
from util.loss_vis import plot_reconstruction_and_error, plot_loss
from data.loading import load_all_data
from data.rdms import create_5D_rdms
import torch

from networks.mlp_emb_kin import MlpEmbKin
from networks.rnn_emb_kin import RnnEmbKin
from networks.cnn_emb_kin import CnnEmbKin

mlp_emb_kin_path = "./models/mlp_emb_kin/001"
rnn_emb_kin_path = "./models/rnn_emb_kin/001"
cnn_emb_kin_path = "./models/cnn_emb_kin/001"

(train_eeg_data, val_eeg_data, test_eeg_data), (
    train_kin,
    val_kin,
    test_kin,
) = load_all_data(EEG_DATA_PATH, KIN_DATA_PATH)

for model, model_path in zip(
    [MlpEmbKin(16), RnnEmbKin(16), CnnEmbKin(16)],
    [mlp_emb_kin_path, rnn_emb_kin_path, cnn_emb_kin_path],
):
    plot_reconstruction_and_error(
        model_path,
        torch.Tensor(val_eeg_data),
        torch.squeeze(torch.Tensor(val_kin)),
        model,
    )
    plot_loss(
        model_path,
        torch.Tensor(val_eeg_data),
        create_5D_rdms(torch.Tensor(val_kin)),
        model,
    )
