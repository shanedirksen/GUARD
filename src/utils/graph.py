from scipy.spatial.distance import cosine
import torch

def add_edges_to_graph(G, t, initial_search_size=15, increase_step=5, max_search_limit=100):
    search_size = initial_search_size
    edges_added = 0

    while edges_added == 0 and search_size <= max_search_limit:
        nodes_data = list(G.nodes(data=True))
        for idx, node_data in enumerate(nodes_data):
            neighbor_indices = t.get_nns_by_item(idx, search_size)

            for neighbor_idx in neighbor_indices:
                if neighbor_idx >= len(nodes_data):
                    print(f"Invalid index {neighbor_idx} for length {len(nodes_data)}")
                    continue

                if neighbor_idx != idx:  # Avoid self-loops
                    similarity = 1.0 - cosine(
                        nodes_data[idx][1]['attr_dict']['latent'].clone().detach().cpu().numpy(),
                        nodes_data[neighbor_idx][1]['attr_dict']['latent'].clone().detach().cpu().numpy())
                    G.add_edge(idx, neighbor_idx,
                               weight=torch.tensor(similarity).clone().detach())
                    edges_added += 1

        if edges_added == 0:
            print(f"No edges added with search size {search_size}. Increasing search size by {increase_step}.")
            search_size += increase_step

    if edges_added == 0:
        print("Forcing connections due to no edges being added.")
        for idx in range(len(G)):
            closest_nodes = t.get_nns_by_item(idx, 1, include_distances=False)
            closest_node = closest_nodes[0]
            if closest_node == idx:
                closest_node = t.get_nns_by_item(idx, 2, include_distances=False)[1]

            G.add_edge(idx, closest_node, weight=1)  # You may decide to use a different weight
            edges_added += 1

    return G, edges_added
