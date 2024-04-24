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

        # The model is updated lazily (https://support.gurobi.com/hc/en-us/articles/19459921918737-What-does-Awaiting-Model-Update-mean)
        self.model.update()

        # Add constraints (we do not need theses constraints for the nodes `o` and `d`)
        for v in list(city.graph.get_vertices())[:-2]:
            # 30b
            self.solver.addConstr(
                quicksum(self.solver.getVarByName(e.name) for e in self.city.graph.get_outgoing_edges(v))
                == quicksum(self.solver.getVarByName(e.name) for e in self.city.graph.get_incoming_edges(v))
            )

            # 30c
            self.solver.addConstr(
                quicksum(self.solver.getVarByName(e.name) for e in self.city.graph.get_outgoing_edges(v)) == 1
            )

            # 30d
            delay_v = None  # TODO: d_v^s (total delay of v in scenario s)
            delay_u = None  # TODO: d_v^s (total delay of v in scenario s)
            gamma_v = None  # TODO: gamma_v^s (intrinsic delay of v in scenario s)
            slack_uv = None  # TODO: s^s_{u,v} (slack between u and v in scenario s)

            for s in self.city.n_scenarios:
                self.solver.addConstr(
                    delay_v
                    >= gamma_v
                    + quicksum(
                        (delay_u - slack_uv) * self.solver.getVarByName(e.name)
                        for e in self.city.graph.get_outgoing_edges(v)
                    )
                )

                # 30e
                self.solver.addConstr(delay_v >= gamma_v)

        # Update manually
        self.solver.update()

    def solve(self):
        # Set objective
        delay_v = None  # TODO: d_v^s (total delay of v in scenario s)
        self.solver.setObjective(
            self.city.delay_cost
            / self.city.n_scenarios
            * quicksum(
                quicksum(delay_v for v in list(self.city.graph.get_vertices())[:-2]) for s in self.city.scenarios
            )
            + self.city.vehicle_cost
            * quicksum(
                self.solver.getVarByName(e.name) for e in self.city.graph.get_outgoing_edges(self.city.graph.source)
            ),
            GRB.MINIMIZE,
        )

        # Optimize solver
        self.solver.optimize()

        return [var.X for var in self.model.getVars()]
