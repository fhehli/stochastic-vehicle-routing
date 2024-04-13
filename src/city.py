import numpy as np

class City:
    def __init__(self, height: int, width: int, n_districts_y: int, n_districts_x: int, n_tasks: int, n_scenarios: int):
        self.height = height
        self.width = width
        self.n_districts_x = n_districts_x
        self.n_districts_y = n_districts_y
        self.n_districts = n_districts_x * n_districts_y
        self.n_tasks = n_tasks
        self.n_scenarios = n_scenarios
        self.sample_tasks(60.0 * 6, 60.0 * 20, 1.2, 1.6)

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

    def sample_congestion(self):
        mu = np.random.uniform(1, 3, 1)
        sigma = np.random.uniform(.4, .6, 1)
        # zeta^district
        congestion = np.random.lognormal(mu, sigma, size=(self.n_districts, 24))
        for i in range(self.n_districts):
            for j in range(1, 24):
                congestion[i, j] = congestion[i, j - 1] / 2 + congestion[i, j]

        mu = 0.02
        sigma = 0.05

        I = np.random.lognormal(mu, sigma, size=1)
        # zeta^inter
        inter_congestion = np.zeros(24)
        inter_congestion[0] = I
        for i in range(1, 24):
            inter_congestion[i] = (inter_congestion[i - 1] + 0.1) * I

        return congestion, inter_congestion    

    def sample_scenarios(self):
        for i in range(self.n_scenarios):
            congestion, inter_congestion = self.sample_congestion()
            # TODO: compute perturbed start and end times as in https://batyleo.github.io/StochasticVehicleScheduling.jl/stable/dataset/#City
            # TODO: find out how to compute the intrinsic delays gamma from this
        
if __name__ == "__main__":
    city = City(50, 50, 5, 5, 10)
    print(city.end_times)  # 55
