import numpy as np
import scipy

from constants import *


class Vertex:
    def __init__(self, name, **kwargs) -> None:
        self.name = name


class Edge:
    def __init__(self, name, from_vertex, to_vertex, **kwargs) -> None:
        self.name = name
        self.from_vertex = from_vertex
        self.to_vertex = to_vertex


class SimpleDirectedGraph:
    # every element in dict maps: name -> {Vertex | Edge}
    # can be empty
    def __init__(self, vertices: dict[Vertex] = None, edges: dict[Edge] = None):
        # perform basic validity check
        self.vertices = dict() if vertices is None else vertices
        self.edges = dict() if edges is None else edges
        self.__validity_check(self.vertices, self.edges)

    def get_vertices(self):
        return self.vertices.values()

    def get_edges(self):
        return self.edges.values()

    def get_num_vertices(self) -> int:
        return len(self.vertices)

    def get_num_edges(self) -> int:
        return len(self.edges)

    def get_vertex_by_name(self, name: str) -> Vertex:
        if name in self.vertices:
            return self.vertices[name]
        else:
            raise NameError(f"Cannot get vertex with name {name}, name does not exist")

    def add_vertex(self, v: Vertex):
        if self.__check_exists_vertex_name(v.name):
            raise NameError(f"Cannot add vertex with name {v.name}, name already exists")
        else:
            self.vertices[v.name] = v

    def add_edge(self, e: Edge):
        if not (
            self.__check_exists_vertex_name(e.from_vertex.name) or self.__check_exists_vertex_name(e.to_vertex.name)
        ):
            raise ValueError(f"cannot add edge with name {e.name}, at least one of the endpoints does not exist")
        if len(self.edges) == 0:
            self.edges[e.name] = e
            self.__validity_check(self.vertices, self.edges)
        else:
            for k, v in self.edges.items():
                if e.name == k:
                    raise NameError(f"cannot add edge with name {e.name}, name already exists")
                if v.from_vertex.name == e.from_vertex.name and v.to_vertex.name == e.to_vertex.name:
                    raise NameError(
                        f"cannot add edge, duplicate edge with same from and to vertex with name {k} al;ready exists"
                    )
            self.edges[e.name] = e
            print(f"add edge {e.from_vertex.name} -> {e.to_vertex.name}")

    def __check_exists_vertex_name(self, name: str) -> bool:
        if len(self.vertices) == 0:
            return False
        else:
            return name in self.vertices

    def __check_exists_edge_name(self, name: str) -> bool:
        if len(self.edges) == 0:
            return True
        else:
            return name in self.edges

    def __validity_check(self, vertices: dict[Vertex], edges: dict[Edge]):
        assert not (
            len(vertices) == 0 and len(edges) > 0
        ), "edges exists but there are no vertices in the graph yet"  # no ghost edges


class City:
    def __init__(self, height: int, width: int, n_districts_y: int, n_districts_x: int, n_tasks: int, n_scenarios: int):
        self.height = height
        self.width = width
        self.n_districts_x = n_districts_x
        self.n_districts_y = n_districts_y
        self.n_districts = n_districts_x * n_districts_y
        self.n_tasks = n_tasks
        self.n_scenarios = n_scenarios

        self.positions_start = None  # n_tasks
        self.positions_end = None
        self.start_times = None
        self.end_times = None
        self.scenario_start_times = None  # n_tasks x n_scenarios
        self.scenario_end_times = None  # n_tasks x n_scenarios
        self.scenario_delays_inter = None  # n_scenarios x (n_districts x 24)
        self.scenario_delays_intra = None  # n_scenarios x (24)
        self.graph = SimpleDirectedGraph()

        self.sample_tasks(
            start_low=TASK_START_TIME_LOW,
            start_high=TASK_START_TIME_HI,
            multiplier_low=TASK_DISTANCE_MULTIPLIER_LOW,
            multiplier_high=TASK_DISTANCE_MULTIPLIER_HI,
        )

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
        city_center = (self.width / 2, self.height / 2)

        positions_start = [(x, y) for (x, y) in zip(x_start, y_start)] + [city_center] * 2
        positions_end = [(x, y) for (x, y) in zip(x_end, y_end)] + [city_center] * 2
        self.positions_start = positions_start
        self.positions_end = positions_end

        final_task_time = N_HOURS * 60.0 - 1
        random_delay = np.random.lognormal(SCENARIO_START_ZERO_UNIFORM_LOW, SCENARIO_START_ZERO_UNIFORM_HI, 2)
        start_times = np.concatenate(([0.0], np.random.uniform(start_low, start_high, self.n_tasks), [final_task_time]))
        multipliers = np.concatenate(
            ([random_delay[0]], np.random.uniform(multiplier_low, multiplier_high, self.n_tasks), [random_delay[1]])
        )
        end_times = start_times + multipliers * np.array(
            [self.distance(x1, y1, x2, y2) for (x1, y1), (x2, y2) in zip(positions_start, positions_end)]
        )
        self.start_times = start_times
        self.end_times = end_times

    # computes the congestion and inter-congestion for every district of the city
    def sample_congestion(self):
        # zeta^district: size n_districts x 24
        hrs = N_HOURS
        mu = np.random.uniform(INTRA_DISTRICT_CONGESTION_MU_UNIF_LOW, INTRA_DISTRICT_CONGESTION_MU_UNIF_HI, 1)
        sigma = np.random.uniform(INTRA_DISTRICT_CONGESTION_SIGMA_UNIF_LO, INTRA_DISTRICT_CONGESTION_SIGMA_UNIF_HI, 1)

        congestion = np.random.lognormal(mu, sigma, size=(self.n_districts, hrs))
        for i in range(self.n_districts):
            for j in range(1, 24):
                congestion[i, j] = congestion[i, j - 1] / 2 + congestion[i, j]

        # zeta^inter: size 24
        mu = INTER_DISTRICT_CONGESTION_MU
        sigma = INTER_DISTRICT_CONGESTION_SIGMA
        J = np.random.lognormal(mu, sigma, size=1)[0]  # numpy 1.26.4

        inter_congestion = np.zeros(hrs)
        inter_congestion[0] = J
        for i in range(1, hrs):
            inter_congestion[i] = (inter_congestion[i - 1] + 0.1) * J

        return congestion, inter_congestion

    # get scenario start and end times and the scenario delay. Each district of the city
    # has its own delay in a specific scenario, i.e. district -> congestion[scenaorio, hr]
    def sample_scenarios(self):
        start_districts = np.array([self.get_district(x, y) for x, y in self.positions_start])
        end_districts = np.array([self.get_district(x, y) for x, y in self.positions_end])
        start_times = self.start_times
        end_times = self.end_times

        # for every task, sample a delay for every scenario
        scenario_start_random_delay = np.random.lognormal(
            SCENARIO_START_ZERO_UNIFORM_LOW, SCENARIO_START_ZERO_UNIFORM_HI, self.n_scenarios
        )
        self.scenario_start_times = np.stack(
            [
                np.repeat(self.start_times[t], self.n_scenarios) + scenario_start_random_delay
                for t in range(self.n_tasks + 2)
            ]
        )
        self.scenario_end_times = np.zeros((self.n_tasks + 2, self.n_scenarios))
        self.scenario_delays_intra = np.zeros((self.n_scenarios, self.n_districts, N_HOURS))
        self.scenario_delays_inter = np.zeros((self.n_scenarios, N_HOURS))

        # sample congestion and inter-congestion for every scenario per task
        for i in range(self.n_scenarios):
            congestion, inter_congestion = self.sample_congestion()
            self.scenario_delays_intra[i, :, :] = congestion
            self.scenario_delays_inter[i, :] = inter_congestion

            for j in range(1, self.n_tasks + 1):
                z1 = self.scenario_start_times[j, i]
                start_district_delay = congestion[start_districts[j], self.get_hour(self.scenario_start_times[j, i])]
                z2 = z1 + start_district_delay
                z3 = z2 + end_times[i] - start_times[i] + inter_congestion[self.get_hour(z2)]
                end_district_delay = congestion[end_districts[j], self.get_hour(z3)]
                self.scenario_end_times[j, i] = z3 + end_district_delay

    # initiates a simple directed graph of the city
    def create_graph(self):
        assert not (
            self.positions_start is None or self.positions_end is None
        ), "cannot create graph with positions not computed"
        assert not (
            self.start_times is None or self.end_times is None
        ), "cannot create graph with start and end times not computed"

        n_verts = self.n_tasks + 2  # [starting_task, ...job_tasks, end_task]
        starting_task = 0
        end_task = n_verts - 1
        job_tasks = range(1, self.n_tasks + 1)
        self.task_routes = []

        # init vertices
        for v in range(n_verts):
            self.graph.add_vertex(Vertex(name=str(v)))

        # build graph for each task
        for origin_id in job_tasks:
            # add every task to base
            self.graph.add_edge(
                Edge(
                    name=f"{starting_task}->{origin_id}",
                    from_vertex=Vertex(str(starting_task)),
                    to_vertex=Vertex(str(origin_id)),
                )
            )
            self.graph.add_edge(
                Edge(
                    name=f"{origin_id}->{end_task}", from_vertex=Vertex(str(origin_id)), to_vertex=Vertex(str(end_task))
                )
            )

            # there is an edge only if we can reach destination from origin before start of task
            for dest_id in range((origin_id + 1), self.n_tasks):
                # travel time from task_i end position to task_j start position
                end_pos_x, end_pos_y = self.positions_end[origin_id]
                start_pos_x, start_pos_y = self.positions_start[dest_id]
                travel_time = self.distance(end_pos_x, end_pos_y, start_pos_x, start_pos_y)
                origin_end_time = self.end_times[origin_id]
                dest_begin_time = self.start_times[dest_id]

                if origin_end_time + travel_time <= dest_begin_time:
                    self.graph.add_edge(
                        Edge(
                            name=f"{origin_id}->{dest_id}",
                            from_vertex=Vertex(str(origin_id)),
                            to_vertex=Vertex(str(dest_id)),
                        )
                    )

    def get_perturbed_travel_time(self, from_node_id: str, to_node_id: str, scenario: int):
        # assumes that the node names are directly convertible to ints
        old_task_id = int(from_node_id)
        new_task_id = int(to_node_id)
        end_pos_x, end_pos_y = self.positions_end[old_task_id]
        start_pos_x, start_pos_y = self.positions_start[new_task_id]
        origin_district = self.get_district(end_pos_x, end_pos_y)
        dest_district = self.get_district(start_pos_x, start_pos_y)

        z1 = self.scenario_end_times[old_task_id, scenario]
        z2 = z1 + self.scenario_delays_intra[scenario, origin_district, self.get_hour(z1)]
        z3 = (
            z2
            + self.distance(end_pos_x, end_pos_y, start_pos_x, start_pos_y)
            + self.scenario_delays_inter[scenario, self.get_hour(z2)]
        )
        result = z3 + self.scenario_delays_intra[scenario, dest_district, self.get_hour(z3)]

        return result

    # computes the slack in minutes for features
    def compute_slacks_for_features(self, from_node_id: str, to_node_id: str) -> np.ndarray:
        # assumes that the node names are directly convertible to ints
        assert self.graph is not None, "cannot compute features with empty graph"

        old_task_id = int(from_node_id)
        new_task_id = int(to_node_id)
        end_pos_x, end_pos_y = self.positions_end[old_task_id]
        start_pos_x, start_pos_y = self.positions_start[new_task_id]

        travel_time = self.distance(end_pos_x, end_pos_y, start_pos_x, start_pos_y)
        perturbed_end_times = self.scenario_end_times[old_task_id, :]
        perturbed_start_times = self.scenario_start_times[new_task_id, :]
        return perturbed_start_times - (perturbed_end_times + travel_time)

    # TODO computes the slacks in minutes for all instances
    def compute_slacks_for_instance(self) -> np.ndarray:
        # assumes that vertex names can be directly converted into ints
        G = self.graph
        E = G.get_edges()
        N = G.get_num_vertices()
        slack_list = np.array(
            [
                [
                    (self.scenario_start_times[int(e.to_vertex.name), s] if int(e.to_vertex.name) < N else np.Inf)
                    - (
                        self.end_times[int(e.from_vertex.name)]
                        + self.get_perturbed_travel_time(int(e.from_node.name), int(e.to_node.name), s)
                    )
                    for s in range(self.n_scenarios)
                ]
                for e in E
            ]
        )
        J = np.array([int(e.from_node.name) for e in E])
        K = np.array([int(e.to_node.name) for e in E])
        return scipy.sparse(J, K, slack_list)  # TODO: check this

    # Returns a matrix of features of size (20, nb_edges)
    def compute_features(self) -> np.ndarray:
        assert self.graph is not None, "cannot compute features with empty graph"

        n_feats = NUM_FEATURES
        features = np.zeros((n_feats, self.graph.get_num_edges()))
        cumul = CUMULATIVE_VALUES_SLACK_EVAL

        # features indices
        travel_time_idx = 0
        connected_to_src_idx = 1
        slack_decile_idxs = range(2, 11)
        slack_cum_distr_idxs = range(11, n_feats)

        # we should be able to index the edges by id so that we can refer edge src/dest id to task id (getting start/end positions)
        for i, edge in enumerate(list(self.graph.get_edges())):
            # compute travel time
            print(len(self.positions_end))
            from_vertex_x, from_vertex_y = self.positions_end[int(edge.from_vertex.name)]
            to_vertex_x, to_vertex_y = self.positions_start[int(edge.to_vertex.name)]
            features[travel_time_idx, i] = self.distance(from_vertex_x, from_vertex_y, to_vertex_x, to_vertex_y)
            features[connected_to_src_idx, i] = VEHICLE_COST if edge.from_vertex.name == 1 else 0.0

            slacks = self.compute_slacks_for_features(int(edge.from_vertex.name), int(edge.to_vertex.name))
            features[slack_decile_idxs, i] = np.quantile(slacks, 0.1 * np.array(range(9)))
            features[slack_cum_distr_idxs, i] = [np.mean(slacks <= x) for x in cumul]

        return features

    # TODO compute delays for instance
    def compute_delays(self):
        d = np.zeros((self.n_tasks, self.n_scenarios))
        self.scenario_end_times - self.end_times

        return d

    @staticmethod
    def get_hour(minutes: float) -> int:
        # assert minutes >= 0, f"Minutes must be positive, got {minutes}"
        # assert minutes <= N_HOURS * 60, f"Minutes must be less equal than {N_HOURS * 60}, got {minutes}"
        return int((minutes % (N_HOURS * 60)) // 60)


if __name__ == "__main__":
    city = City(CITY_HEIGHT_MINUTES, CITY_WIDTH_MINUTES, N_DISTRICTS_X, N_DISTRICTS_Y, N_TASKS, N_SCENARIOS)
    print("non-perturbed start and end times (in minutes): ")
    print(city.start_times)
    print(city.end_times)

    print("mean of perturbed start and end times (in minutes): ")
    print([t for t in np.mean(city.scenario_start_times, axis=1)])
    print([t for t in np.mean(city.scenario_end_times, axis=1)])

    assert np.all(
        [
            np.all(t_end >= t_start)
            for (t_end, t_start) in zip(city.scenario_end_times[:-1], city.scenario_start_times[:-1])
        ]
    )

    print("build graph")
    city.create_graph()
    print("build graph done")

    print("compute features")
    feats = city.compute_features()
    print("compute features done")
    print("features: ", feats)
    print("features shape: ", feats.shape)
