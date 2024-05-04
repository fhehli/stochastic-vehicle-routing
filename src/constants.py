# Define default values here

# For the default values used in the paper:
# see https://github.com/BatyLeo/StochasticVehicleScheduling.jl/blob/main/src/instance/default_values.jl

import numpy as np

# city properties
N_HOURS = 24
CITY_HEIGHT_MINUTES = 50  # city height (in minutes) we assume that the city is squared
CITY_WIDTH_MINUTES = 50
N_DISTRICTS_X = 5  # num districts on the x axis
N_DISTRICTS_Y = 5  # num districts on the y axis
INTRA_DISTRICT_CONGESTION_MU_UNIF_LOW = 1
INTRA_DISTRICT_CONGESTION_MU_UNIF_HI = 3
INTRA_DISTRICT_CONGESTION_SIGMA_UNIF_LO = 0.4
INTRA_DISTRICT_CONGESTION_SIGMA_UNIF_HI = 0.6
INTER_DISTRICT_CONGESTION_MU = 0.02
INTER_DISTRICT_CONGESTION_SIGMA = 0.05
VEHICLE_COST = 1000  # cost for 1 vehicle
DELAY_COST = 2.0  # cost of one minute of delay

# task properties
N_TASKS = 10
N_SCENARIOS = 5
SCENARIO_START_ZERO_UNIFORM_LOW = -np.Inf  # create a LogNormal(mu=-inf, std=1.0), which always returns 0
SCENARIO_START_ZERO_UNIFORM_HI = 1.0
TASK_START_TIME_LOW = 60.0 * 6  # tasks start at 6AM
TASK_START_TIME_HI = 60.0 * 20  # tasks end at 8PM
TASK_DISTANCE_MULTIPLIER_LOW = 1.2  # for drawing random tasks
TASK_DISTANCE_MULTIPLIER_HI = 1.6  # for drawing random tasks

# feature properties
CUMULATIVE_VALUES_SLACK_EVAL = [-100, -50, -20, -10, 0, 10, 50, 200, 500]
NUM_FEATURES = (
    2 + 9 + len(CUMULATIVE_VALUES_SLACK_EVAL)
)  # arc_len in minutes, vehicle cost from source, 9 deciles, slack cumulative prob. distr.
