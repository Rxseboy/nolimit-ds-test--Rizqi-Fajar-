"""
Configuration module for Tweet Sentiment Project
Defines hyperparameters, model identifiers, file paths, and global constants.
"""

import os
from dataclasses import dataclass


@dataclass
class Paths:
	BASE_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
	DATASET_DIR: str = os.path.join(BASE_DIR, "dataset")
	TRAINED_MODEL_DIR: str = os.path.join(BASE_DIR, "trained_model")
	EMBEDDINGS_FILE: str = os.path.join(TRAINED_MODEL_DIR, "train_embeddings.npy")
	KNN_INDEX_FILE: str = os.path.join(TRAINED_MODEL_DIR, "knn_index.pkl")


@dataclass
class ModelConfig:
	MODEL_NAME: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
	MAX_LENGTH: int = 256
	NUM_LABELS: int = 3
	DEVICE: str = "cuda" if os.environ.get("USE_CUDA", "0") == "1" else "cpu"


@dataclass
class TrainingConfig:
	LR: float = 2e-5
	EPOCHS: int = 5
	TRAIN_BATCH: int = 8
	EVAL_BATCH: int = 8
	WEIGHT_DECAY: float = 0.01
	LOGGING_STEPS: int = 50
	OUTPUT_DIR: str = os.path.join(Paths.BASE_DIR, "results")
	LOG_DIR: str = os.path.join(Paths.BASE_DIR, "logs")


paths = Paths()
model_cfg = ModelConfig()
train_cfg = TrainingConfig()