import argparse

# This script can be used to create a training dataset
from city import City
from SVSPSolver import SVSPSolver
from constants import *
import pickle


def main(args):
    # Create the Dataset
    dataset = {"X": [], "Y": []}
    for _ in range(NUM_SAMPLES):
        data = create_datapoint()
        dataset["X"].append(data[0])
        dataset["Y"].append(data[1])

    # Store the created Dataset
    with open(args.out_file, "wb") as out_file:
        pickle.dump(np.array(dataset), out_file)


def create_datapoint(args):
    # Initialize the city
    city = City(
        height=CITY_HEIGHT_MINUTES,
        width=CITY_WIDTH_MINUTES,
        n_districts_x=N_DISTRICTS_X,
        n_districts_y=N_DISTRICTS_Y,
        n_tasks=args.n_tasks,
        n_scenarios=args.n_scenarios,
        seed=args.seed,
    )
    city.create_graph()

    # Initilialize the Gurobi Solver
    solver = SVSPSolver(city)

    # Compute the features used for training the GLM
    glm_features = city.compute_features()  # Matrix of features of size (20, nb_edges)

    # Compute the solution (supervised dataset)
    solution = solver.solve()  # Array with length

    return [glm_features, solution]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=1, help="Number of samples to generate.")
    parser.add_argument("--n_scenarios", type=int, default=N_SCENARIOS, help="Number of scenarios.")
    parser.add_argument("--n_tasks", type=int, default=N_TASKS, help="Number of tasks.")
    parser.add_argument("--out_file", type=str, default=f"data/test.pkl", help="Path to the output file")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")

    args = parser.parse_args()
    main(args)
