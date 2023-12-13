import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
from datetime import datetime
import networkx as nx
from scipy.spatial.distance import cosine
from annoy import AnnoyIndex
from sklearn.model_selection import ParameterGrid
from src.utils.visualization import plot_3d_graph, plot_graph, create_and_save_tree, plot_and_save_communities, create_and_save_multiple_tree
from src.utils.loss import calculate_quartet_loss_from_graph
from src.utils.utils import calculate_avg_class_weights, calculate_avg_family_weights, compute_statistical_thresholds, thresholds
from src.utils.metrics import compute_modularity, evaluate_communities
from src.models.autoencoder import Autoencoder
from src.datasets.dataloader import data_loader
from src.utils.graph import add_edges_to_graph
from collections import Counter
import community as community_module
from collections import defaultdict
from typing import Tuple

def check_for_zero_gradients(model):
    zero_gradient = False
    for param in model.parameters():
        if param.grad is not None and torch.sum(param.grad.abs()) == 0:
            zero_gradient = True
            break
    return zero_gradient

def main(attack_type):
    family_only = False
    X_train, y_train, X_test, y_test, X_hidden, y_hidden, le, unhidden_count, family_map = data_loader([attack_type], "nsl-kdd", family_only)

    print("\nFull list of class labels and their encoded values:")
    total_classes = len(le.classes_)

    for i in range(unhidden_count, total_classes):
        print(f"({i - total_classes}) {le.classes_[i]}")

    for i in range(unhidden_count):
        print(f"({i}) {le.classes_[i]}")


    def print_class_distribution(y_data, label_encoder, title):
        print(f"\n{title} class distribution: ")

        adjusted_indices = [i if i < unhidden_count else i - len(label_encoder.classes_) for i in np.unique(y_data)]

        sorted_indices = sorted(adjusted_indices)

        for class_index in sorted_indices:
            class_count = np.sum(
                y_data == (class_index if class_index >= 0 else class_index + len(label_encoder.classes_)))
            if class_index < 0:
                print(
                    f"({class_index}) {label_encoder.classes_[class_index + len(label_encoder.classes_)]}: {class_count}")
            else:
                print(f"({class_index}) {label_encoder.classes_[class_index]}: {class_count}")


    print_class_distribution(y_train, le, "Training")
    print_class_distribution(y_test, le, "Testing")
    print_class_distribution(y_hidden, le, "Hidden")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device", device)

    param_grid = {
        'lr': [0.00001],
        'batch_size': [100],
        'encoding_dim': [128],
        'optimizer': ['Adam']
    }

    grid = ParameterGrid(param_grid)

    X_train_tensor = torch.Tensor(X_train).to(device)
    y_train_tensor = torch.Tensor(y_train).long().to(device)
    X_test_tensor = torch.Tensor(X_test).to(device)
    y_test_tensor = torch.Tensor(y_test).long().to(device)

    input_dim = X_train.shape[1]

    optimizers = {'Adam': torch.optim.Adam, 'SGD': torch.optim.SGD, 'RMSprop': torch.optim.RMSprop}

    for params in grid:
        print(f"Testing parameters: {params}")

        print("Encoding Dim: ", params['encoding_dim'])
        model = Autoencoder(input_dim, encoding_dim=params['encoding_dim'], num_classes=4).to(device)

        reconstruction_criterion = nn.MSELoss()

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_dataloader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_dataloader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)

        Optimizer = optimizers[params['optimizer']]

        optimizer_encoder = Optimizer(list(model.encoder.parameters()), lr=params['lr'])
        optimizer_decoder = Optimizer(list(model.decoder.parameters()), lr=params['lr'])
        optimizer_modularity = Optimizer(list(model.encoder.parameters()), lr=params['lr'])

        num_epochs = 10

        num_communities = len(np.unique(y_train))
        print("Num Communities: ", num_communities)

        sample_to_track_index = None
        batch_to_track_index = None

        running_average_quartet_loss = 0.0
        running_average_reconstruction_loss = 0.0
        alpha = 0.9  # This is a smoothing factor. It determines the weight of the old average vs the new value.

        for epoch in range(num_epochs):
            model.train()

            batch_sizes = [batch[0].size(0) for batch in train_dataloader]
            print(f"Epoch {epoch + 1} batch sizes: {batch_sizes}")

            total_quartet_loss = 0.0
            total_modularity_score = 0.0
            total_reconstruction_loss = 0.0

            f = params['encoding_dim']
            t = AnnoyIndex(f, 'angular')  # Length of item vector that will be indexed

            pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader),
                        desc=f"Epoch {epoch + 1}/{num_epochs}",
                        dynamic_ncols=True, leave=False)

            for batch_idx, (X_batch, y_batch) in pbar:

                if X_batch.size(0) <= 8:
                    print(f"Skipping batch {batch_idx} of size {X_batch.size(0)}")
                    continue

                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                G = nx.Graph()

                recon, latent_rep, class_probs = model(X_batch)

                reconstruction_loss = reconstruction_criterion(recon, X_batch)
                reconstruction_error = (
                            X_batch - recon).detach()  # Detach to prevent gradients from flowing through this

                f = params['encoding_dim']
                t = AnnoyIndex(f, 'angular')  # Length of item vector that will be indexed

                for latent, error, true_label, original_data in zip(latent_rep, reconstruction_error, y_batch, X_batch):
                    G.add_node(len(G),
                               attr_dict={'latent': latent,
                                          'error': error,
                                          'original_data': original_data,
                                          'predicted_label': None,  # Start with a placeholder value
                                          'true_label': true_label.item()})

                    t.add_item(len(G) - 1, latent.clone().detach().cpu().numpy())

                reconstruction_error_scalar = ((recon - X_batch) ** 2).mean(dim=1)

                if epoch == 0 and sample_to_track_index is None:
                    indices_above_threshold = (reconstruction_error_scalar > 15).nonzero(as_tuple=True)[0]
                    if len(indices_above_threshold) > 0:  # if there's any sample above threshold
                        sample_to_track_index = indices_above_threshold[0].item()  # get the first index
                        batch_to_track_index = batch_idx

                if batch_idx == batch_to_track_index and sample_to_track_index is not None:
                    error_of_tracked_sample = reconstruction_error_scalar[sample_to_track_index].item()

                t.build(10)  # Build the index with 10 trees

                G, edges_added = add_edges_to_graph(G, t)

                if batch_idx == 0 and edges_added > 0:
                    G_copy = G.copy()  # Create a copy of the graph G
                    for node, data in G_copy.nodes(data=True):
                        mean_error = data['attr_dict']['error'].mean().item()
                        data['attr_dict']['error'] = mean_error

                    plot_3d_graph(G_copy, epoch + 1, phase="training")

                t.unbuild()  # Unbuild the index after using it

                sample_G = G.copy()  # Copy G to sample_G for calculating modularity loss
                if len(sample_G.edges()) == 0:
                    print("Sample_G has no edges!")
                    print("Original G edges count:", len(G.edges()))


                quartet_loss = calculate_quartet_loss_from_graph(sample_G, model, device, le, family_map, family_only)
                modularity_score = compute_modularity(sample_G)
                normalized_quartet_loss = quartet_loss / (running_average_quartet_loss + 1e-10)
                normalized_reconstruction_loss = reconstruction_loss / (running_average_reconstruction_loss + 1e-10)
                normalized_quartet_loss.requires_grad_(True)
                normalized_reconstruction_loss.requires_grad_(True)
                optimizer_encoder.zero_grad()
                optimizer_decoder.zero_grad()
                normalized_quartet_loss.backward(retain_graph=True)
                if check_for_zero_gradients(model):
                    print("Warning: Zero gradient detected!")
                optimizer_encoder.step()
                normalized_reconstruction_loss.backward()
                if check_for_zero_gradients(model):
                    print("Warning: Zero gradient detected after reconstruction loss backward pass!")

                optimizer_decoder.step()

                total_quartet_loss += quartet_loss.item()
                total_modularity_score += modularity_score
                total_reconstruction_loss += reconstruction_loss.item()

            average_quartet_loss = total_quartet_loss / len(train_dataloader)
            average_modularity_score = total_modularity_score / len(train_dataloader)
            average_reconstruction_loss = total_reconstruction_loss / len(train_dataloader)

            print(
                f'\nEnd of training epoch {epoch + 1}/{num_epochs}: Average Training Quartet Loss = {average_quartet_loss:.4f}, Average Training Reconstruction Loss = {average_reconstruction_loss:.4f}, Total Reconstruction Loss = {total_reconstruction_loss}, Average Training Modularity Score = {average_modularity_score:.4f}'
            )

            if epoch == num_epochs - 1:

                model.eval()

                G_test = nx.Graph()

                f = params['encoding_dim']
                t_test = AnnoyIndex(f, 'angular')

                total_quartet_loss_test = 0
                total_modularity_score_test = 0
                total_reconstruction_loss_test = 0

                with torch.no_grad():
                    pbar_test = tqdm(enumerate(test_dataloader), total=len(test_dataloader),
                                     desc=f"Epoch {epoch + 1}/{num_epochs}",
                                     dynamic_ncols=True, leave=False)
                    for batch_idx, (X_batch, y_batch) in pbar_test:
                        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                        recon, latent_rep, class_probs = model(X_batch)

                        reconstruction_error = ((recon - X_batch) ** 2).mean(axis=1)
                        reconstruction_loss = reconstruction_criterion(recon, X_batch)

                        for latent, error, true_label, original_data in zip(latent_rep, reconstruction_error, y_batch,
                                                                            X_batch):
                            G_test.add_node(len(G_test),
                                            attr_dict={'latent': latent,
                                                       'error': error,
                                                       'original_data': original_data,
                                                       'predicted_label': None,  # Start with a placeholder value
                                                       'true_label': true_label.item(),
                                                       'hidden': False})  # Add the 'hidden' attribute and set it to False for original nodes

                            t_test.add_item(len(G_test) - 1, latent.clone().detach().cpu().numpy())

                        t_test.build(10)  # 10 trees

                        G_test, edges_added = add_edges_to_graph(G_test, t_test)

                        t_test.unbuild()  # Unbuild the index after using it

                        sample_G_test = G_test.copy()  # Copy G_test to sample_G for calculating modularity loss

                        quartet_loss_test = calculate_quartet_loss_from_graph(sample_G_test, model, device, le, family_map, family_only)
                        total_quartet_loss_test += quartet_loss_test.item()

                        modularity_score_test = compute_modularity(sample_G_test)
                        total_modularity_score_test += modularity_score_test

                        reconstruction_loss_test = (recon - X_batch).pow(2).mean()
                        total_reconstruction_loss_test += reconstruction_loss_test.item()

                        pbar_test.set_postfix(
                            {
                                'Testing Quartet Loss': f'{quartet_loss_test.item():.4f}',
                                'Testing Modularity Score': f'{modularity_score_test:.4f}',
                                'Testing Reconstruction Loss': f'{reconstruction_loss_test.item():.4f}'
                            },
                            refresh=True
                        )

                    average_quartet_loss_test = total_quartet_loss_test / len(test_dataloader)
                    average_modularity_score_test = total_modularity_score_test / len(test_dataloader)
                    average_reconstruction_loss_test = total_reconstruction_loss_test / len(test_dataloader)

                    print(
                        f'End of testing epoch {epoch + 1}/{num_epochs}: Average Testing Quartet Loss = {average_quartet_loss_test:.4f}, Average Testing Reconstruction Loss = {average_reconstruction_loss_test:.4f}, Average Testing Modularity Score = {average_modularity_score_test:.4f}'
                    )

        G_test_known_only = G_test.copy()

        X_hidden_batch = torch.tensor(X_hidden, dtype=torch.float32).to(device)
        recon_hidden, latent_rep_hidden, _ = model(X_hidden_batch)

        reconstruction_error_hidden = ((recon_hidden - X_hidden_batch) ** 2).mean(axis=1)

        hidden_label_offset = total_classes - unhidden_count


        for idx, (hidden_latent, error) in enumerate(zip(latent_rep_hidden, reconstruction_error_hidden)):
            node_id = len(G_test)

            if y_hidden[idx] >= unhidden_count:
                true_label = y_hidden[idx] - total_classes
            else:
                true_label = y_hidden[idx]

            G_test.add_node(node_id,
                            attr_dict={'latent': hidden_latent.cpu(),
                                       'error': error.cpu(),
                                       'predicted_label': None,  # Initialize as None
                                       'true_label': true_label,
                                       'hidden': True})

            t_test.add_item(node_id, hidden_latent.clone().detach().cpu().numpy())

        print("Length of hidden dataset", len(y_hidden))

        negative_label_count = sum(1 for label in y_hidden if label >= unhidden_count)

        print(f"Count of negative labels in hidden dataset: {negative_label_count}")

        unique_labels, counts = np.unique(y_hidden, return_counts=True)
        print(f"Unique class labels in y_hidden_test: {unique_labels}")
        print(f"Counts for each class in y_hidden_test: {counts}")

        t_test.build(10)  # 10 trees

        for node_id in G_test.nodes:
            neighbor_indices = t_test.get_nns_by_item(node_id, 15)

            for neighbor_idx in neighbor_indices:
                if neighbor_idx not in G_test.nodes:
                    print(f"Invalid node ID {neighbor_idx} in G_test")
                    continue

                node_label = G_test.nodes[node_id]['attr_dict']['true_label']
                neighbor_label = G_test.nodes[neighbor_idx]['attr_dict']['true_label']
                if G_test.nodes[node_id]['attr_dict']['hidden'] and G_test.nodes[neighbor_idx]['attr_dict'][
                    'hidden']:
                    continue

                if neighbor_idx != node_id:  # Avoid self-loops
                    similarity = 1.0 - cosine(
                        G_test.nodes[node_id]['attr_dict']['latent'].cpu().detach().numpy(),
                        G_test.nodes[neighbor_idx]['attr_dict']['latent'].cpu().detach().numpy())
                    G_test.add_edge(node_id, neighbor_idx,
                                    weight=torch.tensor(
                                        similarity).cpu())  # Add an edge with weight being the cosine similarity

        t_test.unbuild()  # Unbuild the index after using it

    plot_3d_graph(G_test, epoch + 1, "testing")

    error_mean, error_std, similarity_mean, similarity_std = compute_statistical_thresholds(G_test_known_only)

    print(f"Average Strength of Connections for Known Nodes: {similarity_mean:.4f}")
    print(f"Average Reconstruction Error for Known Nodes: {error_mean:.4f}")

    FP, FN, TP, TN, macro_f1 = thresholds(G_test, error_mean, error_std, similarity_mean, similarity_std)

    rec_unk = TP / (TP + FN) if (TP + FN) != 0 else 0
    precision_unknown = TP / (TP + FP) if (TP + FP) != 0 else 0
    f1_unknown = 2 * (precision_unknown * rec_unk) / (precision_unknown + rec_unk) if (precision_unknown + rec_unk) != 0 else 0

    recall_known = TN / (TN + FP) if (TN + FP) != 0 else 0
    precision_known = TN / (TN + FN) if (TN + FN) != 0 else 0
    f1_known = 2 * (precision_known * recall_known) / (precision_known + recall_known) if (precision_known + recall_known) != 0 else 0

    macro_f1 = (f1_unknown + f1_known) / 2

    print(f"False Positives (known identified as unknown): {FP}")
    print(f"False Negatives (unknown identified as known): {FN}")
    print(f"True Positives (unknown correctly identified): {TP}")
    print(f"True Negatives (known correctly identified): {TN}")
    print(f"Recall-Unknown (Rec-Unk): {rec_unk:.4f}")
    print(f"Precision (Unknown): {precision_unknown:.4f}")
    print(f"Recall (Known): {recall_known:.4f}")
    print(f"Precision (Known): {precision_known:.4f}")
    print(f"Macro F1 Score: {macro_f1:.4f}")

    predicted_unknown_ids = [node_id for node_id, data in G_test.nodes(data=True) if
                             data['attr_dict'].get('predicted_label') == -1]


    plot_3d_graph(G_test_known_only, epoch + 1, "inferencing")
    print(f"Number of predicted unknown nodes: {len(predicted_unknown_ids)}")

    avg_class_weights = calculate_avg_class_weights(predicted_unknown_ids, G_test, le, unhidden_count, family_map)
    most_similar_class = max(avg_class_weights, key=avg_class_weights.get)
    print(f"The known class most similar to the hidden class on average is: {most_similar_class}")

    avg_family_weights = calculate_avg_family_weights(avg_class_weights, family_map)
    most_similar_family = max(avg_family_weights, key=avg_family_weights.get)
    print(f"The known family most similar to the hidden class on average is: {most_similar_family}")

    create_and_save_tree(le, family_map, unhidden_count, epoch + 1, most_similar_family)



    unknown_subgraph = G_test.subgraph(
        predicted_unknown_ids).copy()  # Create a copy to avoid modifying the original graph

    sample_node_id = list(unknown_subgraph.nodes())[0]
    t_test = AnnoyIndex(len(unknown_subgraph.nodes[sample_node_id]['attr_dict']['latent']), 'angular')

    for node_id in unknown_subgraph.nodes():
        t_test.add_item(node_id, unknown_subgraph.nodes[node_id]['attr_dict']['latent'].cpu().detach().numpy())

    t_test.build(10)  # 10 trees

    for node_id in unknown_subgraph.nodes():
        neighbor_indices = t_test.get_nns_by_item(node_id, 15)
        for neighbor_idx in neighbor_indices:
            if neighbor_idx != node_id:  # Avoid self-loops
                similarity = 1.0 - cosine(
                    unknown_subgraph.nodes[node_id]['attr_dict']['latent'].cpu().detach().numpy(),
                    unknown_subgraph.nodes[neighbor_idx]['attr_dict']['latent'].cpu().detach().numpy())
                unknown_subgraph.add_edge(node_id, neighbor_idx, weight=torch.tensor(similarity).cpu())

    t_test.unbuild()  # Unbuild the index after using it

    partitions = community_module.best_partition(unknown_subgraph, resolution=50)

    community_labels = defaultdict(set)
    for node, community_id in partitions.items():
        true_label = unknown_subgraph.nodes[node]['attr_dict']['true_label']
        community_labels[community_id].add(true_label)

    plot_and_save_communities(unknown_subgraph, partitions, community_labels, epoch)

    significant_communities, significant_community_nodes = evaluate_communities(unknown_subgraph, partitions, community_labels,
                                                       predicted_unknown_ids)

    similar_families_list = []

    for community_id, nodes in significant_community_nodes.items():
        avg_class_weights = calculate_avg_class_weights(nodes, G_test, le, unhidden_count,
                                                        family_map)
        most_similar_class = max(avg_class_weights, key=avg_class_weights.get)
        print(f"For Community {community_id}, the most similar class is: {most_similar_class}")

        avg_family_weights = calculate_avg_family_weights(avg_class_weights, family_map)
        most_similar_family = max(avg_family_weights, key=avg_family_weights.get)
        print(f"For Community {community_id}, the most similar family is: {most_similar_family}")

        similar_families_list.append(most_similar_family)

    if len(significant_community_nodes) > 1:
        create_and_save_multiple_tree(le, family_map, unhidden_count, epoch, similar_families_list)

    return most_similar_class, most_similar_family, len(significant_communities), macro_f1

if __name__ == "__main__":
    family_map = {
        'back': 'DoS', 'land': 'DoS', 'neptune': 'DoS', 'pod': 'DoS', 'smurf': 'DoS', 'teardrop': 'DoS',
        'mailbomb': 'DoS', 'processtable': 'DoS', 'udpstorm': 'DoS', 'apache2': 'DoS',
        'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe', 'satan': 'Probe', 'mscan': 'Probe', 'saint': 'Probe',
        'ftp_write': 'R2L', 'guess_passwd': 'R2L', 'imap': 'R2L', 'multihop': 'R2L', 'phf': 'R2L', 'spy': 'R2L',
        'warezclient': 'R2L', 'warezmaster': 'R2L', 'sendmail': 'R2L', 'named': 'R2L', 'snmpgetattack': 'R2L',
        'snmpguess': 'R2L', 'xlock': 'R2L', 'xsnoop': 'R2L', 'worm': 'R2L',
        'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R', 'rootkit': 'U2R', 'httptunnel': 'U2R',
        'ps': 'U2R', 'sqlattack': 'U2R', 'xterm': 'U2R'
    }

    correct_class_predictions = 0
    correct_family_predictions = 0
    total_multi_attack_checks = 0
    correct_multi_attack_predictions = 0
    f1_scores = []

    attack_types = list(family_map.keys())

    for attack in attack_types:
        print(f"\nProcessing attack type: {attack}")
        predicted_class, predicted_family, detected_communities, f1_score = main([attack])  # Wrap attack in a list

        class_accuracy = 1 if family_map.get(predicted_class) == family_map.get(attack) else 0
        family_accuracy = 1 if predicted_family == family_map.get(attack) else 0
        correct_class_predictions += class_accuracy
        correct_family_predictions += family_accuracy
        f1_scores.append(f1_score)

        if detected_communities == 1:  # Comparing detected communities against the expected count of 1
            correct_multi_attack_predictions += 1
        total_multi_attack_checks += 1

        print(
            f"Class Match: {class_accuracy}, Family Match: {family_accuracy}, "
            f"Correct Multi-Attack Detection: {detected_communities == 1}, F1 Score: {f1_score}"
        )

    overall_class_accuracy = correct_class_predictions / len(attack_types)
    overall_family_accuracy = correct_family_predictions / len(attack_types)
    multi_attack_accuracy = correct_multi_attack_predictions / total_multi_attack_checks
    avg_f1_score = sum(f1_scores) / len(attack_types)

    print("\nOverall Class Match Accuracy:", overall_class_accuracy)
    print("Overall Family Match Accuracy:", overall_family_accuracy)
    print("Overall Multi-Attack Detection Accuracy:", multi_attack_accuracy)
    print("Average F1 Score:", avg_f1_score)
