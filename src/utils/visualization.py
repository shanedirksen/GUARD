import os
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import seaborn as sns
import torch
from networkx.drawing.nx_pydot import graphviz_layout
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pyvis.network import Network

def generate_color_map(unknown_subgraph):
    unique_labels = set(node_data['attr_dict']['true_label'] for _, node_data in unknown_subgraph.nodes(data=True))

    colormap = plt.cm.jet  # You can choose any other colormap like plt.cm.viridis, plt.cm.plasma, etc.
    num_labels = len(unique_labels)
    colors = [colormap(i) for i in np.linspace(0, 1, num_labels)]

    color_map = dict(zip(unique_labels, colors))

    return color_map


def plot_and_save_communities(unknown_subgraph, partitions, community_labels, epoch):
    main_dir = os.path.join('outputs', 'community')
    if not os.path.exists(main_dir):
        os.makedirs(main_dir)

    color_map = generate_color_map(unknown_subgraph)

    community_shapes = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'H']
    node_shapes = {community: shape for community, shape in zip(community_labels.keys(), community_shapes)}

    layout = nx.spring_layout(unknown_subgraph)
    plt.figure(figsize=(12, 12))

    sorted_class_labels = sorted(color_map.keys())
    sorted_communities = sorted(node_shapes.keys())

    label_patches = [Patch(color=color_map[label], label=f"Class {label}") for label in sorted_class_labels]
    shape_patches = [
        Line2D([0], [0], marker=node_shapes[community], color='w', markerfacecolor='gray', markersize=10,
               label=f"Community {community}") for community in sorted_communities]

    for community, shape in node_shapes.items():  # <-- Corrected this line
        nx.draw_networkx_nodes(unknown_subgraph, pos=layout,
                               nodelist=[node for node in unknown_subgraph.nodes() if partitions[node] == community],
                               node_color=[color_map[unknown_subgraph.nodes[node]['attr_dict']['true_label']] for node
                                           in
                                           unknown_subgraph.nodes() if partitions[node] == community],
                               node_shape=shape)

    nx.draw_networkx_edges(unknown_subgraph, pos=layout)

    plt.legend(handles=label_patches + shape_patches, loc='upper right')

    plt.savefig(os.path.join(main_dir, f'community_epoch_{epoch + 1}.png'))


def evaluate_communities(unknown_subgraph, partitions, community_labels, predicted_unknown_ids):
    community_sizes = {
        community_id: len([node for node in unknown_subgraph.nodes() if partitions[node] == community_id])
        for community_id in community_labels}

    threshold_size = 0.25 * len(predicted_unknown_ids)

    significant_communities = [community_id for community_id, size in community_sizes.items() if size > threshold_size]

    print(f"Number of significant communities (potential hidden classes): {len(significant_communities)}")

    for community_id in significant_communities:
        print(f"Community {community_id} size: {community_sizes[community_id]}")
        labels_list = [unknown_subgraph.nodes[node]['attr_dict']['true_label'] for node in unknown_subgraph.nodes() if
                       partitions[node] == community_id]
        class_counts = Counter(labels_list)
        print(f"Class counts for Community {community_id}:")
        for class_label, count in class_counts.items():
            print(f"Class {class_label}: {count}")

def plot_graph(G, epoch_num):
    graphs_dir = os.path.join('outputs', 'graphs')

    if not os.path.exists(graphs_dir):
        os.makedirs(graphs_dir)

    seed = 42  # You can choose any number you want.
    pos = nx.spring_layout(G, seed=seed)
    plt.figure(figsize=(10, 10))

    label_dict = {0: 0, 1: 1, 2: 2, 3: 3, 'unknown': 4}

    color_values = [label_dict[data['attr_dict']['true_label']] for _, data in G.nodes(data=True)]

    nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('viridis'),
                           node_color=color_values,
                           alpha=0.8)
    nx.draw_networkx_edges(G, pos, alpha=0.2)

    plt.savefig(os.path.join(graphs_dir, f'graph_{epoch_num}.png'), format="PNG")
    plt.close()

def plot_3d_graph(G, epoch_num, phase="training"):
    """
    Visualizes a 3D graph of node embeddings, reconstruction errors, and the inherent structure.

    Args:
    - G: The graph to visualize.
    - epoch_num: Current epoch number (for file naming).
    - label_encoder: The label encoder used to transform class labels.
    - phase: Either 'training' or 'testing'. Determines the subfolder to save the graph image.
    """
    assert phase in ["training", "testing", "inferencing"], "Phase should be either 'training', 'testing', or 'inferencing."

    main_dir = os.path.join('outputs', '3dgraphs')
    sub_dir = os.path.join(main_dir, phase)

    if not os.path.exists(main_dir):
        os.makedirs(main_dir)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)

    node_data = [data['attr_dict'] for _, data in G.nodes(data=True)]
    reconstruction_errors = np.array([data['error'].item() if torch.is_tensor(data['error']) else data['error'] for data in node_data])

    pos = nx.spring_layout(G, seed=42)
    positions = np.array(list(pos.values()))

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    unique_labels = sorted(list(set(data['true_label'] for data in node_data)))

    num_hidden_classes = sum(1 for label in unique_labels if label < 0)
    hidden_palette = [(i / (num_hidden_classes * 2), i / (num_hidden_classes * 2), i / (num_hidden_classes * 2)) for i
                      in reversed(range(num_hidden_classes))]

    unhidden_count = len(unique_labels) - num_hidden_classes
    palette = hidden_palette + list(sns.color_palette("hls", unhidden_count))

    color_map = {label: palette[i] for i, label in enumerate(unique_labels)}
    color_values = [color_map[data['true_label']] for data in node_data]

    scatter = ax.scatter(positions[:, 0], positions[:, 1], reconstruction_errors, c=color_values, alpha=0.8)

    node_mapping = {node: idx for idx, node in enumerate(G.nodes())}

    for edge in G.edges():
        x = np.array([positions[node_mapping[edge[0]], 0], positions[node_mapping[edge[1]], 0]])
        y = np.array([positions[node_mapping[edge[0]], 1], positions[node_mapping[edge[1]], 1]])
        z = np.array([reconstruction_errors[node_mapping[edge[0]]], reconstruction_errors[node_mapping[edge[1]]]])
        ax.plot(x, y, z, color='grey', alpha=0.2)

    legend_labels = [
        plt.Line2D([0], [0], marker='o', color='w', label=str(label), markersize=10,
                   markerfacecolor=color_map[label]) for label in unique_labels]

    ax.legend(handles=legend_labels, title="Classes")

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_xlabel('Layout X')
    ax.set_ylabel('Layout Y')
    ax.set_zlabel('Reconstruction Error')

    plt.savefig(os.path.join(sub_dir, f'graph_3d_{epoch_num}.png'), format="PNG")
    plt.close()


def create_and_save_tree(label_encoder, family_map, unhidden_count, epoch, most_similar_family=None):
    G = nx.DiGraph()

    G.add_node("Network Activity")

    G.add_node("Normal")
    G.add_edge("Network Activity", "Normal")

    G.add_node("Attacks")
    G.add_edge("Network Activity", "Attacks")

    total_classes = len(label_encoder.classes_)
    hidden_classes = label_encoder.classes_[unhidden_count:total_classes]

    for attack_class, family in family_map.items():
        if attack_class in hidden_classes:
            continue

        if family != attack_class:
            if not G.has_node(family):
                G.add_node(family)
                G.add_edge("Attacks", family)

            G.add_node(attack_class)
            G.add_edge(family, attack_class)
        else:
            G.add_node(attack_class)
            G.add_edge("Attacks", attack_class)

    if most_similar_family:
        G.add_node("unknown class")
        G.add_edge(most_similar_family, "unknown class")

    nt = Network(notebook=True, height="750px", width="100%")
    nt.from_nx(G)

    nt.show_buttons(filter_=["physics"])
    nt.set_options("""
    var options = {
      "nodes": {
        "color": "skyblue"
      },
      "edges": {
        "color": {
          "inherit": true
        },
        "smooth": {
          "type": "continuous"
        }
      },
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -30000,
          "centralGravity": 0.3,
          "springLength": 95,
          "springConstant": 0.04,
          "damping": 0.09,
          "avoidOverlap": 0.1
        },
        "minVelocity": 0.75
      }
    }
    """)

    tree_dir = os.path.join('outputs', 'tree', 'interactive')

    if not os.path.exists(tree_dir):
        os.makedirs(tree_dir)

    output_file = os.path.join(tree_dir, f'tree_{epoch}.html')
    nt.show(output_file)


def create_and_save_multiple_tree(label_encoder, family_map, unhidden_count, epoch, similar_families_list):
    G = nx.DiGraph()

    G.add_node("Network Activity")

    G.add_node("Normal")
    G.add_edge("Network Activity", "Normal")

    G.add_node("Attacks")
    G.add_edge("Network Activity", "Attacks")

    total_classes = len(label_encoder.classes_)
    hidden_classes = label_encoder.classes_[unhidden_count:total_classes]

    for attack_class, family in family_map.items():
        if attack_class in hidden_classes:
            continue

        if family != attack_class:
            if not G.has_node(family):
                G.add_node(family)
                G.add_edge("Attacks", family)

            G.add_node(attack_class)
            G.add_edge(family, attack_class)
        else:
            G.add_node(attack_class)
            G.add_edge("Attacks", attack_class)

    counter = 1
    for most_similar_family in similar_families_list:
        G.add_node(f"unknown class {counter}")
        G.add_edge(most_similar_family, f"unknown class {counter}")
        counter += 1

    pos = graphviz_layout(G, prog="dot")

    plt.figure(figsize=(24, 24))
    nx.draw(G, pos, with_labels=True, node_size=4000, node_color="skyblue", node_shape="o", alpha=0.6,
            linewidths=4)
    plt.title("Network Activity Tree")

    tree_dir = os.path.join('outputs', 'tree', 'multiple')

    if not os.path.exists(tree_dir):
        os.makedirs(tree_dir)

    plt.savefig(os.path.join(tree_dir, f'tree_{epoch}.png'), format="PNG")
    plt.close()
