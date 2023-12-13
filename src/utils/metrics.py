import networkx as nx
from collections import Counter

def evaluate_communities(unknown_subgraph, partitions, community_labels, predicted_unknown_ids):
    community_sizes = {community_id: len([node for node in unknown_subgraph.nodes() if partitions[node] == community_id])
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

    significant_community_nodes = {
        community_id: [node for node in unknown_subgraph.nodes() if partitions[node] == community_id]
        for community_id in significant_communities
    }
    return significant_communities, significant_community_nodes

def compute_modularity(G):
    nodes_data = G.nodes(data=True)
    unique_true_labels_in_graph = set(data['attr_dict']['true_label'] for node, data in nodes_data)
    communities = [set([node for node, data in nodes_data if data['attr_dict']['true_label'] == i]) for i in unique_true_labels_in_graph]

    try:
        modularity_score = nx.algorithms.community.modularity(G, communities)
    except nx.algorithms.community.quality.NotAPartition:
        print("Not a valid partition:")
        print("Graph nodes count:", len(G.nodes()))
        print("Communities count:", len(communities))
        all_nodes_in_communities = set().union(*communities)
        print("Number of unique nodes in communities:", len(all_nodes_in_communities))
        flattened_communities = [node for community in communities for node in community]
        print("Total number of nodes in communities:", len(flattened_communities))
        modularity_score = 0  # or some other default value

    return modularity_score
