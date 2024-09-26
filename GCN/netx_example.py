import networkx as nx 
import matplotlib.pyplot as plt 

G = nx.DiGraph()
G.add_nodes_from([
    (0, {"color": "Blue", "size": 250}),
    (1, {"color": "Green", "size": 350}),
    (2, {"color": "Red", "size": 150}),
    (3, {"color": "Orange", "size": 200}),
])

G.add_edges_from(
    [
        (0,1),
        (1,2),
        (1,0),
        (1,3),
        (2,3),
        (3,0)
    ]
)

for node in G.nodes(data=True):
    print(f"-->Node: {node}")
    
node_colors = nx.get_node_attributes(G, "color").values()
colors = list(node_colors)
print(f"colors: {colors}")
node_sizes = nx.get_node_attributes(G, "size").values()
sizes = list(node_sizes)
nx.draw(G, with_labels=True, node_color=colors, node_size=sizes)
plt.waitforbuttonpress()
