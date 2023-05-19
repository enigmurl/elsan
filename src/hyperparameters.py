from statistics import NormalDist

# pvalue generation
DATA_DIR = "../data/"
DATA_FRAME_SIZE = 63

CON_LIST = [-2, -1, -0.75, -0.5, -0.25, -0.15, 0, 0.15, 0.25, 0.5, 0.75, 1, 2]
P_VALUES = [NormalDist().cdf(c) for c in CON_LIST]

DATA_COUNT_IN_ENSEMBLE = 63
DATA_QUERIES_PER_ENSEMBLE = 8
DATA_NUM_ENSEMBLES = 256
DATA_SKIP_FRAME = 8
DATA_OUT_FRAME = 32

DATA_VALIDATION_ENSEMBLES = 32

CONFIDENCE_SEPARATOR_POWER = 0.15

DATA_STD_SCALE = 6
DATA_NOISE = 0.05
DATA_T_STEP = 0.2
DATA_NU = 3e-4

DATA_LOWER_UNKNOWN_VALUE = -5.0
DATA_UPPER_UNKNOWN_VALUE = 5.0
DATA_LOWER_QUERY_VALUE = 5.0
DATA_UPPER_QUERY_VALUE = -5.0

WARMUP_SLOPE = 4

# clipping
CLIPPING_EPOCHS = 36
CLIPPING_BATCH_SIZE = 32

# model
O_PRUNING_SIZE = 16
O_INPUT_LENGTH = 16
O_KERNEL_SIZE = 3
O_DROPOUT_RATE = 0
O_TIME_RANGE = 1

# training
O_MAX_EPOCH = 1024
O_BATCH_SIZE = 8
O_RUN_SIZE = 64
O_LEARNING_RATE = 1e-3


C_BATCH_SIZE = 128

O_TRAIN_DIREC = "../data/ensemble/"
O_TRAIN_INDICES = list(range(0, 512))
O_MAX_ENSEMBLE_COUNT = 16
C_TRAIN_DIREC = "../data/clipping/"
C_TRAIN_INDICES = list(range(0, int(2 ** 15)))

# metrics
V_BATCH_SIZE = 16
V_DIREC = "../data/validate/"
V_INDICES = list(range(0, DATA_VALIDATION_ENSEMBLES))
