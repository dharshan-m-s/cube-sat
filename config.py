DATASET_PATH = "dataset"

IMG_SIZE = (224, 224)

BATCH_SIZE = 32

EPOCHS = 20
NUM_EPOCHS = 20

MODEL_NAME = "disaster_model.h5"
MODEL_PATH = "output/natural_disaster.model"

CLASSES = ["cyclone", "earthquake", "flood", "wildfire"]

# Data split ratios
TEST_SPLIT = 0.20
VAL_SPLIT = 0.20

# Learning rate parameters
MIN_LR = 1e-7
MAX_LR = 1e-2
STEP_SIZE = 8

# Cyclic Learning Rate method
CLR_METHOD = "triangular"

# Output paths
LRFIND_PLOT_PATH = "output/lrfind_plot.png"
TRAINING_PLOT_PATH = "output/training_plot.png"
CLR_PLOT_PATH = "output/clr_plot.png"
