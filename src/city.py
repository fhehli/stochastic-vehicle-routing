import numpy as np
from constants import *

class City:
    def __init__(self, height: int, width: int, n_districts_y: int, n_districts_x: int, n_tasks: int, n_scenarios: int):
        self.height = height
        self.width = width
        self.n_districts_x = n_districts_x
        self.n_districts_y = n_districts_y
        self.n_districts = n_districts_x * n_districts_y
        self.n_tasks = n_tasks
        self.n_scenarios = n_scenarios

        self.positions_start = None
        self.positions_end = None
        self.start_times = None
        self.end_times = None
        self.scenario_start_times = None # n_tasks x n_scenarios
        self.scenario_end_times = None   # n_tasks x n_scenarios

        self.sample_tasks(start_low=TASK_START_TIME_LOW, 
                          start_high=TASK_START_TIME_HI, 
                          multiplier_low=TASK_DISTANCE_MULTIPLIER_LOW, 
                          multiplier_high=TASK_DISTANCE_MULTIPLIER_HI)
        
        self.sample_scenarios()

    def position_valid(self, x: float, y: float) -> bool:
        # Returns True if the position (x, y) is within the city boundaries
        return 0 <= x <= self.width and 0 <= y <= self.height

    def distance(self, x1: float, y1: float, x2: float, y2: float) -> float:
        assert self.position_valid(x1, y1), f"Position ({x1}, {y1}) is not within the city boundaries"
        assert self.position_valid(x2, y2), f"Position ({x2}, {y2}) is not within the city boundaries"
        # Returns the Euclidean distance between (x1, y1) and (x2, y2)
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def get_district(self, x: float, y: float) -> int:
        assert self.position_valid(x, y), f"Position ({x}, {y}) is not within the city boundaries"
        # Returns the district number of the district at position (x, y)
        return int(y / self.height * self.n_districts_y) * self.n_districts_x + int(x / self.width * self.n_districts_x)

    def sample_tasks(self, start_low: float, start_high: float, multiplier_low: float, multiplier_high: float):
        x_start = np.random.uniform(0, self.width, self.n_tasks)
        y_start = np.random.uniform(0, self.height, self.n_tasks)
        x_end = np.random.uniform(0, self.width, self.n_tasks)
        y_end = np.random.uniform(0, self.height, self.n_tasks)
        positions_start = [(x, y) for (x, y) in zip(x_start, y_start)]
        positions_end = [(x, y) for (x, y) in zip(x_end, y_end)]
        self.positions_start = positions_start
        self.positions_end = positions_end
        start_times = np.random.uniform(start_low, start_high, self.n_tasks)
        multipliers = np.random.uniform(multiplier_low, multiplier_high, self.n_tasks)
        end_times = start_times + multipliers * np.array([self.distance(x1, y1, x2, y2) for (x1, y1), (x2, y2) in zip(positions_start, positions_end)])
        self.start_times = start_times
        self.end_times = end_times

    # computes the congestion and inter-congestion for every district of the city
    def sample_congestion(self):
        # zeta^district: size n_districts x 24
        hrs = N_HOURS
        mu = np.random.uniform(INTRA_DISTRICT_CONGESTION_MU_UNIF_LOW, 
                               INTRA_DISTRICT_CONGESTION_MU_UNIF_HI, 
                               1)
        sigma = np.random.uniform(INTRA_DISTRICT_CONGESTION_SIGMA_UNIF_LO, 
                                  INTRA_DISTRICT_CONGESTION_SIGMA_UNIF_HI, 
                                  1)
        
        congestion = np.random.lognormal(mu, sigma, size=(self.n_districts, hrs))
        for i in range(self.n_districts):
            for j in range(1, 24):
                congestion[i, j] = congestion[i, j - 1] / 2 + congestion[i, j]

        # zeta^inter: size 24
        mu = INTER_DISTRICT_CONGESTION_MU
        sigma = INTER_DISTRICT_CONGESTION_SIGMA
        I = np.random.lognormal(mu, sigma, size=1)[0] # numpy 1.26.4

        inter_congestion = np.zeros(hrs)
        inter_congestion[0] = I
        for i in range(1, hrs):
            inter_congestion[i] = (inter_congestion[i - 1] + 0.1) * I

        return congestion, inter_congestion    

    def sample_scenarios(self):
        start_districts = np.array([self.get_district(x, y) for x, y in self.positions_start])
        end_districts = np.array([self.get_district(x, y) for x, y in self.positions_end])
        start_times = self.start_times
        end_times = self.end_times

        # for every task, sample a delay for every scenario
        scenario_start_random_delay = np.random.lognormal(SCENARIO_START_ZERO_UNIFORM_LOW, 
                                                         SCENARIO_START_ZERO_UNIFORM_HI, 
                                                         self.n_scenarios)
        self.scenario_start_times = np.stack([np.repeat(self.start_times[t], self.n_scenarios) + scenario_start_random_delay
                                              for t in range(self.n_tasks)])
        self.scenario_end_times = np.zeros((self.n_tasks, self.n_scenarios))

        # sample congestion and inter-congestion for every scenario
        for j in range(self.n_tasks):
            for i in range(self.n_scenarios):
                congestion, inter_congestion = self.sample_congestion()
                z1 = self.scenario_start_times[j, i]
                start_district_delay = congestion[start_districts[j], self.get_hour(self.scenario_start_times[j, i])] 
                z2 = z1 + start_district_delay
                z3 = z2 + end_times[i] - start_times[i] + inter_congestion[self.get_hour(z2)]
                end_district_delay = congestion[end_districts[j], self.get_hour(z3)]
                self.scenario_end_times[j, i] = z3 + end_district_delay

    # TODO
    def compute_features():
        pass

    # TODO
    def compute_slacks():
        pass    

    @staticmethod
    def get_hour(minutes: float) -> int:
        assert minutes >= 0, f"Minutes must be positive, got {minutes}"
        assert minutes < N_HOURS * 60, f"Minutes must be less than {N_HOURS * 60}, got {minutes}"
        return int(minutes // 60)
        

if __name__ == "__main__":
    city = City(CITY_HEIGHT_MINUTES, 
                CITY_WIDTH_MINUTES, 
                N_DISTRICTS_X, 
                N_DISTRICTS_Y, 
                N_TASKS, 
                N_SCENARIOS)
    print(city.start_times)
    print(city.end_times)
    print([t for t in np.mean(city.scenario_start_times, axis=1)]) 
    print([t for t in np.mean(city.scenario_end_times, axis=1)]) 
    assert np.all([np.all(t_end > t_start) for (t_end, t_start) in zip(city.scenario_end_times, city.scenario_start_times)])
