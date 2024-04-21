# Gurobi Solver
import gurobipy as gp
from gurobipy import GRB
from gurobipy import quicksum


class VSPSolver:
    def __init__(self, city):
        # Store the city
        self.city = city

        # Create the Gurobi Solver instance
        self.model = gp.Model("VSP")

        # Create the linear variables for the Solver (one for each edge)
        for edge in self.city.graph.get_edges():
            self.model.addVar(vtype=GRB.BINARY, name=edge.name)

        # Add constraints
        for v in city.graph.get_vertices():
            # Flow Polytope
            self.model.addConstr(
                quicksum(self.model.getVarByName(e.name) for e in city.graph.get_outgoing_edges(v))
                == quicksum(self.model.getVarByName(e.name) for e in city.graph.get_incoming_edges(v))
            )
            # Task Covering
            self.model.addConstr(
                quicksum(self.model.getVarByName(e.name) for e in city.graph.get_outgoing_edges(v)) == 1
            )

        # The model is updated lazily (https://support.gurobi.com/hc/en-us/articles/19459921918737-What-does-Awaiting-Model-Update-mean)
        # Update manually
        self.model.update()

    def solve(self, theta):
        # Set objective
        self.model.setObjective(theta.dot(self.model.getVars()), GRB.MAXIMIZE)

        # Optimize model
        self.model.optimize()

        # TODO: maybe need to extract values (solution stored in `var.X`)
        # TODO: what do we return in case when no solution is found
        return self.model.getVars()
