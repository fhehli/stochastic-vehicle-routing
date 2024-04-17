# Default values
import numpy as np

# city properties
N_HOURS = 24
CITY_HEIGHT_MINUTES = 50
CITY_WIDTH_MINUTES = 50
N_DISTRICTS_X = 5
N_DISTRICTS_Y = 5
INTRA_DISTRICT_CONGESTION_MU_UNIF_LOW = 1
INTRA_DISTRICT_CONGESTION_MU_UNIF_HI = 3
INTRA_DISTRICT_CONGESTION_SIGMA_UNIF_LO = .4
INTRA_DISTRICT_CONGESTION_SIGMA_UNIF_HI = .6
INTER_DISTRICT_CONGESTION_MU = 0.02
INTER_DISTRICT_CONGESTION_SIGMA = 0.05
VEHICLE_COST = 10 # TODO: find out actual number used

# task properties
N_TASKS = 10
N_SCENARIOS = 5
SCENARIO_START_ZERO_UNIFORM_LOW = -np.Inf
SCENARIO_START_ZERO_UNIFORM_HI = 1.
TASK_START_TIME_LOW = 60.0 * 6
TASK_START_TIME_HI = 60.0 * 20
TASK_DISTANCE_MULTIPLIER_LOW = 1.2
TASK_DISTANCE_MULTIPLIER_HI = 1.6

# feature properties
CUMULATIVE_VALUES_SLACK_EVAL = [-100, -50, -20, -10, 0, 10, 50, 200, 500]
NUM_FEATURES = 2 + 9 + len(CUMULATIVE_VALUES_SLACK_EVAL) # arc_len in minutes, vehicle cost from source, 9 deciles, slack cumulative prob. distr.