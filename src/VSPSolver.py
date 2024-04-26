# Gurobi Solver
import gurobipy as gp
from gurobipy import GRB
from gurobipy import quicksum


# Appendix B.1.1 of https://arxiv.org/pdf/2207.13513.pdf
# https://github.com/BatyLeo/StochasticVehicleScheduling.jl/blob/main/src/learning/easy_problem.jl
class VSPSolver:
    def __init__(self, city):
        # Store the city
        self.city = city

        # Create the Gurobi Solver instance
        self.model = gp.Model("VSP")

        # Create the linear variables for the Solver (one for each edge)
        for edge in self.city.graph.get_edges():
            self.model.addVar(vtype=GRB.BINARY, name=edge.name)

        # The model is updated lazily (https://support.gurobi.com/hc/en-us/articles/19459921918737-What-does-Awaiting-Model-Update-mean)
        self.model.update()

        # Add constraints (we do not need theses constraints for the nodes `o` and `d`)
        for v in list(self.city.graph.get_vertices())[:-2]:
            # Flow Polytope
            self.model.addConstr(
                quicksum(self.model.getVarByName(e.name) for e in self.city.graph.get_outgoing_edges(v))
                == quicksum(self.model.getVarByName(e.name) for e in self.city.graph.get_incoming_edges(v))
            )
            # Task Covering
            self.model.addConstr(
                quicksum(self.model.getVarByName(e.name) for e in self.city.graph.get_outgoing_edges(v)) == 1
            )

        # Update manually
        self.model.update()

    def solve(self, theta):
        # Set objective
        self.model.setObjective(theta.dot(self.model.getVars()), GRB.MINIMIZE)

        # Optimize model
        self.model.optimize()

        return [var.X for var in self.model.getVars()]