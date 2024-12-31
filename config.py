import os

# Configuration parameters
DATA_DIR = "imagenet-dataset"   # Default data directory
NUM_EPOCHS = 20                 # Number of training epochs
BATCH_SIZE = 512              # Batch size for training
LEARNING_RATE = 0.5            # Learning rate for the optimizer

# Create the data directory if it does not exist
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    print(f"Created data directory: {DATA_DIR}")
