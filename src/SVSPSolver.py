from itertools import product
import numpy as np

# Gurobi Solver
import gurobipy as gp
from gurobipy import GRB
from gurobipy import quicksum


# Appendix B.1.2 (MILP formulation) of https://arxiv.org/pdf/2207.13513.pdf
# https://github.dev/BatyLeo/StochasticVehicleScheduling.jl/blob/main/src/solution/exact_algorithms/plne.jl
class SVSPSolver:
    """Implements a MILP model for the stochastic VSP using the Gurobi Python API

    Attributes:
        city: The class containing the SVSP
        model: The instance of the Gurobi model
    """

    def __init__(self, city):
        # Store the city
        self.city = city

        # Create the Gurobi Solver instance
        self.model = gp.Model("SVSP")
        self.model.setParam("OutputFlag", 0)

        # Create the linear variables for the Solver
        # One for each edge
        for edge in self.city.graph.get_edges():
            self.model.addVar(vtype=GRB.BINARY, name=edge.name)

        # One for the node delays per scenario
        for s, v in product(range(self.city.n_scenarios), self.city.graph.get_vertices()):
            self.model.addVar(lb=0, name=f"d({v.name},{s})")

        # Needed for the Mc Cormick linearization constraints
        for s, e in product(range(self.city.n_scenarios), self.city.graph.get_edges()):
            self.model.addVar(lb=0, name=f"yd({e.name},{s})")

        # The model is updated lazily (https://support.gurobi.com/hc/en-us/articles/19459921918737-What-does-Awaiting-Model-Update-mean)
        self.model.update()

        # Add constraints (we do not need theses constraints for the nodes `o` and `d`)
        for v in list(city.graph.get_vertices())[:-2]:
            # 30b
            self.model.addConstr(
                quicksum(self.model.getVarByName(e.name) for e in self.city.graph.get_incoming_edges(v))
                == quicksum(self.model.getVarByName(e.name) for e in self.city.graph.get_outgoing_edges(v))
            )

            # 30c
            self.model.addConstr(
                quicksum(self.model.getVarByName(e.name) for e in self.city.graph.get_incoming_edges(v)) == 1
            )

            delays = self.city.compute_delays()  # matrix[n_scenarios, n_vertices]
            slacks = self.city.compute_slacks_for_instance()  # matrix[node_source, node_dest, scenario]

            for s in range(self.city.n_scenarios):
                # 30d
                self.model.addConstr(
                    self.model.getVarByName(f"d({v.name},{s})")
                    >= delays[s, int(v.name)]
                    + quicksum(
                        self.model.getVarByName(f"yd({e.name},{s})")
                        - self.model.getVarByName(e.name) * slacks[int(e.from_vertex.name), int(e.to_vertex.name), s]
                        for e in self.city.graph.get_incoming_edges(v)
                    )
                )

                # 30e
                self.model.addConstr(self.model.getVarByName(f"d({v.name},{s})") >= delays[s, int(v.name)])

        # Mc Cormcik linearization constraints
        max_delay = np.max(np.sum(delays, axis=0))

        for s, e in product(range(self.city.n_scenarios), self.city.graph.get_edges()):
            self.model.addConstr(
                self.model.getVarByName(f"yd({e.name},{s})")
                >= delays[s, int(e.from_vertex.name)] + max_delay * (self.model.getVarByName(e.name) - 1)
            )
            self.model.addConstr(
                self.model.getVarByName(f"yd({e.name},{s})") <= max_delay * self.model.getVarByName(e.name)
            )

        # Update manually
        self.model.update()

    def solve(self):
        # Set objective
        self.model.setObjective(
            self.city.delay_cost
            / self.city.n_scenarios
            * quicksum(
                quicksum(self.model.getVarByName(f"d({v.name},{s})") for v in list(self.city.graph.get_vertices())[:-2])
                for s in range(self.city.n_scenarios)
            )
            + self.city.vehicle_cost
            * quicksum(
                self.model.getVarByName(e.name)
                for e in self.city.graph.get_outgoing_edges(self.city.graph.get_source())
            ),
            GRB.MINIMIZE,
        )

        # Optimize model
        self.model.optimize()

        # TODO: which variables do we need to return, only the selected edges or also the delays
        return np.array([self.model.getVarByName(e.name).X for e in self.city.graph.get_edges()])
