# Gurobi Solver
import gurobipy as gp
from gurobipy import GRB
from gurobipy import quicksum


class SVSPSolver:
    """Implements a MILP solver for the stochastic VSP using the Gurobi Python API

    Attributes:
        city: The class containing the SVSP
        solver: The instance of the Gurobi solver
    """

    def __init__(self, city):
        # Store the city
        self.city = city

        # Create the Gurobi Solver instance
        self.solver = gp.Model("SVSP")

        # Create the linear variables for the Solver (one for each edge)
        for edge in self.city.graph.get_edges():
            self.solver.addVar(vtype=GRB.BINARY, name=edge.name)

        # Add constraints
        for v in city.graph.get_vertices():
            # Flow Polytope
            self.solver.addConstr(
                quicksum(self.solver.getVarByName(e.name) for e in city.graph.get_outgoing_edges(v))
                == quicksum(self.solver.getVarByName(e.name) for e in city.graph.get_incoming_edges(v))
            )
            # Task Covering
            self.solver.addConstr(
                quicksum(self.solver.getVarByName(e.name) for e in city.graph.get_outgoing_edges(v)) == 1
            )

        # The solver is updated lazily (https://support.gurobi.com/hc/en-us/articles/19459921918737-What-does-Awaiting-Model-Update-mean)
        # Update manually
        self.solver.update()

    def solve(self, theta):
        # Set objective
        self.solver.setObjective(theta.dot(self.solver.getVars()), GRB.MAXIMIZE)

        # Optimize solver
        self.solver.optimize()

        # TODO: maybe need to extract values (solution stored in `var.X``)
        # TODO: what do we return in case when no solution is found
        return self.solver.getVars()
