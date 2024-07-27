from CONFIG import *

from main import train_kl_only_models

train_kl_only_models(
    SEED, EEG_DATA_PATH, KIN_DATA_PATH, EPOCHS, LEARNING_RATE, ALPHA_EMBEDDER
)
