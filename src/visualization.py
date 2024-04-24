import networkx as nx
import matplotlib.pyplot as plt

from city import *
from constants import *


def custom_graph_to_nxgraph(G: SimpleDirectedGraph):
    # nx.DiGraph accepts a list of tuples
    E = G.get_edges()
    edge_list = [(int(e.from_vertex.name), int(e.to_vertex.name)) for e in E]
    nxgraph = nx.DiGraph(edge_list)
    return nxgraph


def visualize_tasks(nxgraph: nx.DiGraph):
    # TODO: add edge attributes

    pos = nx.spring_layout(nxgraph)
    nx.draw(nxgraph, pos, with_labels=True, node_size=1500, node_color="skyblue", font_size=10, arrows=True)

    # labels = nx.get_edge_attributes(nxgraph, 'duration')
    # nx.draw_networkx_edge_labels(nxgraph, pos, edge_labels=labels)

    # for node, (time, location) in nxgraph.nodes.items():
    #     plt.text(pos[node][0], pos[node][1], f"({time})", ha='center', va='center')

    # plt.text(0.8, 0.9, f"Total time: {max([time for time, _ in nxgraph.nodes])}", ha='center', va='center', transform=plt.gca().transAxes)
    plt.title("Tasks Visualization")
    plt.show()


if __name__ == "__main__":
    city = City(CITY_HEIGHT_MINUTES, CITY_WIDTH_MINUTES, N_DISTRICTS_X, N_DISTRICTS_Y, N_TASKS, N_SCENARIOS)
    start_times = city.start_times
    end_times = city.end_times
    city.create_graph()

    G = custom_graph_to_nxgraph(city.graph)
    visualize_tasks(G)
