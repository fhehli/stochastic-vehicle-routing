import gurobipy as gp
from gurobipy import GRB
from gurobipy import quicksum
import torch


# Appendix B.1.1 of https://arxiv.org/pdf/2207.13513.pdf
# https://github.com/BatyLeo/StochasticVehicleScheduling.jl/blob/main/src/learning/easy_problem.jl
def solve_vsp(thetas, graph):
    # Create the Gurobi Solver instance
    model = gp.Model("VSP")
    model.setParam("OutputFlag", 0)

    # Create the linear variables for the Solver (one for each edge)
    for edge in graph.get_edges():
        model.addVar(vtype=GRB.BINARY, name=edge.name)

    # The model is updated lazily (https://support.gurobi.com/hc/en-us/articles/19459921918737-What-does-Awaiting-Model-Update-mean)
    model.update()

    # Add constraints (we do not need theses constraints for the nodes `o` and `d`)
    for v in list(graph.get_vertices())[:-2]:
        # Flow Polytope
        model.addConstr(
            quicksum(model.getVarByName(e.name) for e in graph.get_incoming_edges(v))
            == quicksum(model.getVarByName(e.name) for e in graph.get_outgoing_edges(v))
        )
        # Task Covering
        model.addConstr(quicksum(model.getVarByName(e.name) for e in graph.get_incoming_edges(v)) == 1)

    # Update manually
    model.update()

    solution = torch.empty_like(thetas)

    for i, theta in enumerate(thetas):
        # Set objective
        model.setObjective(quicksum(t * v for t, v in zip(theta, model.getVars())), GRB.MINIMIZE)

        # Clear the previous solution
        model.reset(clearall=0)

        # Optimize model
        model.optimize()

        solution[i] = torch.tensor([var.X for var in model.getVars()])

    return solution
