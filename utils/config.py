# image and mask parameters
ORIGINAL_IMAGE_SIZE = (256, 256)
IMAGE_SIZE = (32, 32)  # a param to reconsider resolution so different
IMAGE_SHAPE = IMAGE_SIZE + (3,)  # 3 channels for RGB
NUM_MASKS = 2  # binary classification (forest vs not forest)

# CNN parameters
KERNEL_SIZE = (3, 3)
POOL_SIZE = (2, 2)
FILTERS_NUM = 32

# training parameters
EPOCHS = 12
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
BUFFER_SIZE = 1000

ORIGINAL_RESOLUTION = 0.5
RESOLUTION = ORIGINAL_RESOLUTION * ORIGINAL_IMAGE_SIZE[0]/IMAGE_SIZE[0]

# directories
DATASET_DIR = "C:\\Users\\jjasi\\datasets\\ForestSegmented"
TRAINED_MODEL_DIR = f"trained_models/res{RESOLUTION}-lr{LEARNING_RATE}-n_flt{FILTERS_NUM}eph{EPOCHS}"
