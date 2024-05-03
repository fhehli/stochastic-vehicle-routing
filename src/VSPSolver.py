from torch import tensor

# Gurobi Solver
import gurobipy as gp
from gurobipy import GRB
from gurobipy import quicksum


# Appendix B.1.1 of https://arxiv.org/pdf/2207.13513.pdf
# https://github.com/BatyLeo/StochasticVehicleScheduling.jl/blob/main/src/learning/easy_problem.jl
class VSPSolver:
    def solve(self, theta, graph):
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

        # Set objective
        model.setObjective(quicksum(t * v for t, v in zip(theta, model.getVars())), GRB.MINIMIZE)

        # Optimize model
        model.optimize()

        return tensor([var.X for var in model.getVars()])
