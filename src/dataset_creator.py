# This script can be used to create a training dataset
from city import City
from SVSPSolver import SVSPSolver
from constants import *
import pickle


def main():
    # Number of datapoints to generate
    NUM_SAMPLES = 1

    # Create the Dataset
    dataset = {"X": [], "Y": []}
    for _ in range(NUM_SAMPLES):
        data = create_datapoint()
        dataset["X"].append(data[0])
        dataset["Y"].append(data[1])

    # Store the created Dataset
    with open("../data/test.pkl", "wb") as out_file:
        pickle.dump(np.array(dataset), out_file)


def create_datapoint():
    # Initialize the city
    city = City(
        CITY_HEIGHT_MINUTES,
        CITY_WIDTH_MINUTES,
        N_DISTRICTS_X,
        N_DISTRICTS_Y,
        N_TASKS,
        N_SCENARIOS,
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
    main()
