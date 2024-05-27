import pickle
import timeit

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.models import FenchelYoungGLM
from src.SVSPSolver import SVSPSolver
from src.VSPSolver import solve_vsp
from src.utils import CitiesDataset


def get_dataloader(num_instances):
    path = "data/default.pkl"
    with open(path, "rb") as file:
        data = pickle.load(file)
        X = data["X"][:num_instances]
        Y = data["Y"][:num_instances]
        cities = data["cities"][:num_instances]

    dataset = CitiesDataset(X, Y, cities)
    dataloader = DataLoader(dataset, batch_size=None, shuffle=True)

    return dataloader


def milp(solver):
    return solver.solve()


def ml(model, features, graph):
    return solve_vsp(model(features).unsqueeze(0), graph)


n_cities = 1
runs_per_city = 1
milp_times = []
ml_times = []

model = FenchelYoungGLM(n_features=20)
model.load_state_dict(torch.load("models/linear/epoch20.pt"))
model.eval()

dataloader = get_dataloader(n_cities)
with torch.inference_mode():
    for i, (features, label, city) in tqdm(enumerate(dataloader)):
        solver = SVSPSolver(city)

        def milp_wrapper():
            return milp(solver)

        milp_times.append(timeit.timeit(milp_wrapper, number=runs_per_city))

        graph = city.graph

        def ml_wrapper():
            return ml(model, features, graph)

        ml_times.append(timeit.timeit(ml_wrapper, number=runs_per_city))

with open("data/runtime/runtime.csv", "w") as file:
    file.write("milp,ml\n")
    for milp_time, ml_time in zip(milp_times, ml_times):
        file.write(f"{milp_time},{ml_time}\n")

with open("data/runtime/runtime", "w") as file:
    file.write(f"MILP mean     {np.mean(milp_times):.2f}\n")
    file.write(f"MILP std      {np.std(milp_times):.2f}\n")
    file.write("\n")
    file.write(f"ML mean       {np.mean(ml_times):.2f}\n")
    file.write(f"ML std        {np.std(ml_times):.2f}\n")
