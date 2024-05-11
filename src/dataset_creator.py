import argparse
import pickle

# This script can be used to create a training dataset
from src.city import City
from src.SVSPSolver import SVSPSolver
from src.constants import *


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def create_datapoint(args):
    # Initialize the city
    city = City(
        height=CITY_HEIGHT_MINUTES,
        width=CITY_WIDTH_MINUTES,
        n_districts_x=N_DISTRICTS_X,
        n_districts_y=N_DISTRICTS_Y,
        n_tasks=args.n_tasks,
        n_scenarios=args.n_scenarios,
    )
    city.create_graph()

    # Initilialize the Gurobi Solver
    solver = SVSPSolver(city)

    # Compute the features used for training the GLM
    glm_features = city.compute_features()  # Matrix of features of size (20, nb_edges)

    # Compute the solution (supervised dataset)
    solution = solver.solve()  # Array with length

    return glm_features, solution, city


def main(args):
    X, Y, cities = [], [], []
    for _ in range(args.n_samples):
        x, y, city = create_datapoint(args)
        X.append(x)
        Y.append(y)
        cities.append(city)
    
    with open(args.out_file, "wb") as out_file:
        if args.city:
            data = {"X": X, "Y": Y, "cities": cities, "args": vars(args)}
        else:
            graphs = list(map(lambda city: city.graph, cities))
            data = {"X": X, "Y": Y, "graphs": graphs, "args": vars(args)}
        pickle.dump(data, out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=1, help="Number of samples to generate.")
    parser.add_argument("--n_scenarios", type=int, default=N_SCENARIOS, help="Number of scenarios.")
    parser.add_argument("--n_tasks", type=int, default=N_TASKS, help="Number of tasks.")
    parser.add_argument("--city", type=str2bool, nargs='?', const=True, default=False, help="Dump the city instance related to the sample")
    parser.add_argument("--out_file", type=str, default="data/test.pkl", help="Path to the output file")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")

    args = parser.parse_args()
    np.random.seed(args.seed)
    main(args)
