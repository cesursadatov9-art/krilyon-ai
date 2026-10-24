"""Configuration for the KrilyonNEX NEX-1 project."""

PROJECT_NAME = "KrilyonNEX"
MODEL_NAME = "NEX-1"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

# Training and inference settings
MAX_CONTEXT_LENGTH = 1024
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.7
TOP_P = 0.9

# Runtime settings
LOAD_IN_4BIT = False
USE_GPU_IF_AVAILABLE = True

# Dataset paths
TRAIN_DATA_PATH = "data/az_train.txt"
