from CONFIG import SEED
from util.type_hints import *

from training.basic_loops import classic_training_loop
from sklearn.decomposition import PCA, FastICA


def ana_ica(
    eeg_data_training: NDArray, kin_rdms: NDArray, fitting_data: NDArray
) -> Tuple[float, NDArray]:
    """Computes ICA on the passed EEG data and returns a training loss and RDMs based on the passed RDMs and embeddings respectively.
    @param eeg_data_training: The EEG data to be trained on. The data is expected to be of 5 dimensions: (Participants x Grasp phase x Condition x Channels x Time Points)
    @param kin_rdms: The target RDMs.
    @return: A training loss value and training RDMs.
    """
    return classic_training_loop(
        FastICA(
            n_components=1,
            algorithm="parallel",
            whiten="unit-variance",
            max_iter=1000,
            tol=1e-5,
            random_state=SEED,
        ),
        eeg_data_training,
        kin_rdms,
        fitting_data,
    )


def ana_pca(
    eeg_data_training: NDArray, kin_rdms: NDArray, fitting_data: NDArray
) -> Tuple[float, NDArray]:
    """Computes PCA on the passed EEG data and returns a training loss and RDMs based on the passed RDMs and embeddings respectively.
    @param eeg_data_training: The EEG data to be trained on. The data is expected to be of 5 dimensions: (Participants x Grasp phase x Condition x Channels x Time Points)
    @param kin_rdms: The target RDMs.
    @return: A training loss value and training RDMs.
    """

    return classic_training_loop(
        PCA(n_components=1, copy=True, whiten=False, svd_solver="full"),
        eeg_data_training,
        kin_rdms,
        fitting_data,
    )
