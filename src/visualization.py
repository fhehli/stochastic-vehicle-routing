import networkx as nx
from pyvis.network import Network

import matplotlib.collections as pltcollections
import matplotlib.pyplot as plt 
import mplcursors

from city import *
from constants import *

# returns a 2d-array containing the coordinates of the 2d scatterplot
def get_plt_scatter_coords(pc: pltcollections.PathCollection) -> list:
    return pc.get_offsets()

# TODO
def city_to_nxgraph(city: City) -> nx.Graph:
    # nx.DiGraph accepts a list of tuples
    # we can directly turn nxGrpah into pyviz graph

    # get city properties
    city.create_graph()
    G = city.graph()
    E = G.get_edges()
    V = G.get_vertices()
    n_scenarios = city.n_scenarios
    n_tasks = city.n_tasks
    task_start_positions = city.positions_start
    task_end_positions = city.positions_end

    # create nxgraph attributes
    G_task = nx.DiGraph()
    G_scenario = [nx.DiGraph() for _ in range(n_scenarios)] # one set of edges for each scenario
    edge_list = [(int(e.from_vertex.name), int(e.to_vertex.name)) for e in E]
    node_list = [int(v.name) for v in V] 

    # create task graph
    G_task.add_node()

    return G_task

# TODO
def visualize_graphs(nxgraph: nx.DiGraph, city: City):
        # we should be able to view each chain of tasks individually with start and end times and total time used on the graph
        pos = nx.spring_layout(nxgraph)
        nx.draw(nxgraph, pos, with_labels=True, node_size=400, node_color="skyblue", font_size=10, arrows=True)

        pos = nx.spring_layout(nxgraph)
        nx.draw(nxgraph, pos, with_labels=True, node_size=1500, node_color="skyblue", font_size=10, arrows=True)

    # labels = nx.get_edge_attributes(nxgraph, 'duration')
    # nx.draw_networkx_edge_labels(nxgraph, pos, edge_labels=labels)

    # for node, (time, location) in nxgraph.nodes.items():
    #     plt.text(pos[node][0], pos[node][1], f"({time})", ha='center', va='center')


def visualize_tasks(city: City):
    task_start_positions = city.positions_start
    task_end_positions = city.positions_end
    n_tasks = city.n_tasks

    # Unzip start and end positions
    start_x, start_y = zip(*task_start_positions)
    end_x, end_y = zip(*task_end_positions)
    
    # colors
    cm = plt.get_cmap('gist_rainbow')
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    colors = [cm(1.*i/(n_tasks+2)) for i in range(n_tasks+2)]
    ax.set_prop_cycle(color=colors)

    # figures and legend
    start_scatter = ax.scatter(start_x, start_y, color=colors)
    end_scatter = ax.scatter(end_x, end_y, color=colors, marker='x')

    for start, end, color in zip(task_start_positions, task_end_positions, colors):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        ax.arrow(start[0], start[1], dx, dy, color=color, head_width=0.75, length_includes_head=True)
    ax.legend()

    # labels
    plt.title('Start and End Positions')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    
    # Show plot
    plt.grid(True)
    mplcursors.cursor([start_scatter, end_scatter], hover=True)
    plt.show()


     
if __name__ == "__main__":  
    city = City(CITY_HEIGHT_MINUTES, CITY_WIDTH_MINUTES, N_DISTRICTS_X, N_DISTRICTS_Y, N_TASKS, N_SCENARIOS)
    # G = city_to_nxgraph(city)
    visualize_tasks(city)
