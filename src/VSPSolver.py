import numpy as np

# Gurobi Solver
import gurobipy as gp
from gurobipy import GRB
from gurobipy import quicksum

class VSPSolver:
    def __init__(self, nodes, graph, jobs):
        self.model = gp.Model("VSP")
        # Create variables
        self.graph = {}
        for i in range(nodes):
            for j in range(nodes):
                if graph.has_edge(i, j):
                    self.graph[i, j] = self.model.addVar(vtype=GRB.BINARY)

        # Add constraints
        for i in jobs:
            # Flow Polytope
            self.model.addConstr(
                quicksum(self.graph[j, i] for j in graph.predecessors(i)) == quicksum(self.graph[i, j] for j in graph.successors(i))
            )
            # Task Covering
            self.model.addConstr(
                quicksum(self.graph[j, i] for j in graph.predecessors(i)) == 1
            )

    def solve(self, theta):
        # Set objective
        self.model.setObjective(
            theta.dot(self.paths),
            GRB.MAXIMIZE
        )

        # Optimize model
        self.model.optimize()

        return self.model.getVars() # TODO: maybe need to extract values
