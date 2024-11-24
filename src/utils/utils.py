import numpy as np
import matplotlib.pyplot as plt

def calculate_avg_class_weights(predicted_unknown_ids, G_test_known_only, le, unhidden_count, family_map):
    class_weights_sum = {}
    class_counts = {}

    hidden_offset = le.classes_.shape[0] - unhidden_count

    for node_id in predicted_unknown_ids:
        for neighbor, attributes in G_test_known_only[node_id].items():
            if not G_test_known_only.nodes[neighbor]['attr_dict']['hidden']:
                neighbor_label = G_test_known_only.nodes[neighbor]['attr_dict']['true_label']

                if neighbor_label >= unhidden_count:
                    neighbor_label -= hidden_offset

                class_name = le.inverse_transform([neighbor_label])[0]

                if class_name not in family_map:
                    continue

                class_weights_sum[class_name] = class_weights_sum.get(class_name, 0.0) + attributes['weight'].item()
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

    avg_class_weights = {
        cls: class_counts[cls] / sum(1 / weight for weight in class_weights_sum[cls]) 
        if class_weights_sum[cls] else 0
        for cls in class_weights_sum
    }


    weighted_scores = {}
    for cls, avg_weight in avg_class_weights.items():
        count = class_counts[cls]
        if avg_weight + count != 0:  # Avoid division by zero
            weighted_scores[cls] = 2 * (avg_weight * count) / (avg_weight + count)
        else:
            weighted_scores[cls] = 0

    return weighted_scores

def calculate_avg_family_weights(avg_class_weights, family_map):
    family_weights_sum = {}
    family_counts = {}

    for cls, weight in avg_class_weights.items():
        if cls not in family_map:
            continue

        family = family_map[cls]

        family_weights_sum[family] = family_weights_sum.get(family, 0.0) + weight
        family_counts[family] = family_counts.get(family, 0) + 1

    avg_family_weights = {
        family: family_counts[family] / sum(1 / weight for weight in weights if weight > 0)
        if weights else 0
        for family, weights in family_weights.items()
    }

    return avg_family_weights



def compute_statistical_thresholds(G):
    errors = [data['attr_dict']['error'].item() for node_id, data in G.nodes(data=True)]
    similarities = [G[node_id][neighbor]['weight'].item() for node_id in G.nodes for neighbor in
                    G.neighbors(node_id)]

    error_mean = np.mean(errors)
    error_std = np.std(errors)
    similarity_mean = np.mean(similarities)
    similarity_std = np.std(similarities)

    return error_mean, error_std, similarity_mean, similarity_std

def thresholds(G_test, error_mean, error_std, similarity_mean, similarity_std):
    true_labels = []

    zscore = 1

    hidden_nodes = [(node_id, data) for node_id, data in G_test.nodes(data=True) if data['attr_dict'].get('hidden')]

    for _, data in hidden_nodes:
        true_label = data['attr_dict']['true_label']
        true_labels.append(true_label)

    avg_strength_hidden = np.mean([np.mean([G_test[node_id][neighbor]['weight'].item() for neighbor in G_test.neighbors(node_id)] or [0]) for node_id, _ in hidden_nodes])
    avg_error_hidden = np.mean([data['attr_dict']['error'].item() for _, data in hidden_nodes])

    print(f"Average Connection Strength for Hidden Nodes: {avg_strength_hidden:.4f}")
    print(f"Average Reconstruction Error for Hidden Nodes: {avg_error_hidden:.4f}")

    z_values = [-3, -2, -1, 0, 1, 2, 3]
    error_z_values = [error_mean + (z * error_std) for z in z_values]
    strength_z_values = [similarity_mean + (z * similarity_std) for z in z_values]

    print("Z-score values for Reconstruction Error:")
    for z, value in zip(z_values, error_z_values):
        print(f"Z = {z}: {value:.4f}")

    print("\nZ-score values for Connection Strength:")
    for z, value in zip(z_values, strength_z_values):
        print(f"Z = {z}: {value:.4f}")

    degrees_unknown = []
    degrees_known = []

    for node_id, data in hidden_nodes:
        degree = len(list(G_test.neighbors(node_id)))
        if data['attr_dict']['true_label'] == -1:
            degrees_unknown.append(degree)
        else:
            degrees_known.append(degree)

    degrees_all = degrees_unknown + degrees_known

    mean_all_degrees = sum(degrees_all) / len(degrees_all)
    std_all_degrees = np.std(degrees_all)

    z_values_degrees = [-3, -2, -1, 0, 1, 2, 3]
    degrees_z_values = [mean_all_degrees + (z * std_all_degrees) for z in z_values_degrees]

    print("\nMean number of connections for all hidden nodes:", mean_all_degrees)
    print("Z-score values for Number of Connections:")
    for z, value in zip(z_values_degrees, degrees_z_values):
        print(f"Z = {z}: {value:.2f}")

    avg_degree_unknown = sum(degrees_unknown) / len(degrees_unknown) if degrees_unknown else 0
    avg_degree_known = sum(degrees_known) / len(degrees_known) if degrees_known else 0

    print(f"Average number of connections for true unknown (-1) nodes: {avg_degree_unknown:.2f}")
    print(f"Average number of connections for known (not -1) nodes: {avg_degree_known:.2f}")

    for _, data in hidden_nodes:
        true_label = data['attr_dict']['true_label']
        true_labels.append(true_label)

    z_degrees = [(degree - mean_all_degrees) / std_all_degrees for degree in degrees_all]

    for (node_id, data), z_degree in zip(hidden_nodes, z_degrees):
        edge_weights = [G_test[node_id][neighbor]['weight'].item() for neighbor in G_test.neighbors(node_id)]
        mean_similarity = np.mean(edge_weights) if edge_weights else 0

        z_error = (data['attr_dict']['error'].item() - error_mean) / error_std
        z_similarity = (mean_similarity - similarity_mean) / similarity_std  # Change here

        z_error = z_error if z_error > 0 else 0

        z_similarity = abs(z_similarity) if z_similarity < 0 else 0
        z_degree = abs(z_degree) if z_degree < 0 else 0

        if z_error + (z_similarity * 2) + (z_degree * 2) > zscore:
            predicted_label = -1
        else:
            predicted_label = 0

        data['attr_dict']['predicted_label'] = predicted_label  # Add prediction as an attribute to the node

    FP = sum(1 for node_id, true in zip([node_id for node_id, _ in hidden_nodes], true_labels) if
             true >= 0 and G_test.nodes[node_id]['attr_dict']['predicted_label'] == -1)
    TP = sum(1 for node_id, true in zip([node_id for node_id, _ in hidden_nodes], true_labels) if
             true < 0 and G_test.nodes[node_id]['attr_dict']['predicted_label'] == -1)
    FN = sum(1 for node_id, true in zip([node_id for node_id, _ in hidden_nodes], true_labels) if
             true < 0 and G_test.nodes[node_id]['attr_dict']['predicted_label'] != -1)
    TN = sum(1 for node_id, true in zip([node_id for node_id, _ in hidden_nodes], true_labels) if
             true >= 0 and G_test.nodes[node_id]['attr_dict']['predicted_label'] != -1)

    false_negatives = [(node_id, data) for node_id, data in hidden_nodes if
                       data['attr_dict']['true_label'] < 0 and data['attr_dict']['predicted_label'] == 0]

    for node_id, data in false_negatives:
        edge_weights = [G_test[node_id][neighbor]['weight'].item() for neighbor in G_test.neighbors(node_id)]
        mean_similarity = np.mean(edge_weights) if edge_weights else 0
        degree = len(edge_weights)  # Number of connections for the node

        z_error_node = (data['attr_dict']['error'].item() - error_mean) / error_std
        z_similarity_node = (mean_similarity - similarity_mean) / similarity_std
        z_degree_node = (degree - mean_all_degrees) / std_all_degrees  # This line was missing

        z_similarity_node = abs(z_similarity_node) if z_similarity_node < 0 else 0
        z_error_node = z_error_node if z_error_node > 0 else 0
        z_degree_node = abs(z_degree_node) if z_degree_node < 0 else 0

        total_z = z_error_node + z_similarity_node + z_degree_node

        print(
            f"Node ID: {node_id}, Mean Similarity: {mean_similarity:.4f} (Z: {(z_similarity_node * 2):.2f}), Reconstruction Error: {data['attr_dict']['error'].item():.4f} (Z: {z_error_node:.2f}), Number of Connections: {degree} (Z: {(z_degree_node * 2):.2f}), Total Z: {total_z:.2f}")

    hidden_classes = set([label for label in true_labels if label < 0])
    for hidden_class in hidden_classes:
        correct_predictions = sum(1 for node_id, true in zip([node_id for node_id, _ in hidden_nodes], true_labels) if
                                  true == hidden_class and G_test.nodes[node_id]['attr_dict']['predicted_label'] == -1)
        total_hidden_class = TP + FN  # This is the total number of actual unknowns for class -1
        accuracy = correct_predictions / total_hidden_class if total_hidden_class != 0 else 0
        print(f"Accuracy for class {hidden_class}: {accuracy:.2f}")

    precision_unk = TP / (TP + FP) if TP + FP != 0 else 0
    recall_unk = TP / (TP + FN) if TP + FN != 0 else 0
    f1_unk = 2 * precision_unk * recall_unk / (
            precision_unk + recall_unk) if precision_unk + recall_unk != 0 else 0

    precision_known = TN / (TN + FN) if TN + FN != 0 else 0
    recall_known = TN / (TN + FP) if TN + FP != 0 else 0
    f1_known = 2 * precision_known * recall_known / (
            precision_known + recall_known) if precision_known + recall_known != 0 else 0

    macro_f1 = (f1_unk + f1_known) / 2

    return FP, FN, TP, TN, macro_f1

