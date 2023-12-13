import torch
import torch.nn.functional as F

def pairwise_distances(latent_rep):
    """Compute pairwise distances in the batch."""
    dists = torch.norm(latent_rep[:, None] - latent_rep, dim=2, p=2)
    return dists

def sample_quartets(latent_rep, y_batch, le, family_map, device):
    """Sample quartets."""
    n = latent_rep.size(0)
    distances = pairwise_distances(latent_rep).to(device)

    positive_mask = (y_batch.view(n, 1) == y_batch.view(1, n)).to(device)

    family_id_map = {family: idx for idx, family in enumerate(family_map.values())}
    family_id_map['normal'] = len(family_id_map)

    family_batch = torch.tensor([family_id_map[family_map.get(le.inverse_transform([label.item()])[0], 'normal')] for label in y_batch]).to(device)
    semi_positive_mask = (family_batch.view(n, 1) == family_batch.view(1, n)) & ~positive_mask
    negative_mask = ~(positive_mask | semi_positive_mask)

    anchors, positives, semi_positives, negatives = [], [], [], []
    for i in range(n):
        valid_positives = distances[i, positive_mask[i]]
        valid_semi_positives = distances[i, semi_positive_mask[i]]
        valid_negatives = distances[i, negative_mask[i]]

        if len(valid_positives) == 0 or len(valid_semi_positives) == 0 or len(valid_negatives) == 0:
            continue

        max_positive_dist = valid_positives.max()

        semi_valid_semi_positives = valid_semi_positives[valid_semi_positives < max_positive_dist]

        if len(semi_valid_semi_positives) == 0:
            continue

        min_semi_valid_semi_positive = semi_valid_semi_positives.min()

        valid_hard_negatives = valid_negatives[valid_negatives < min_semi_valid_semi_positive]

        if len(valid_hard_negatives) == 0:
            continue

        min_valid_hard_negative = valid_hard_negatives.min()

        anchors.append(latent_rep[i])
        positives.append(
            latent_rep[torch.nonzero(positive_mask[i], as_tuple=True)[0][valid_positives == max_positive_dist][0]])
        semi_positives.append(
            latent_rep[torch.nonzero(semi_positive_mask[i], as_tuple=True)[0][valid_semi_positives == min_semi_valid_semi_positive][0]])
        negatives.append(
            latent_rep[torch.nonzero(negative_mask[i], as_tuple=True)[0][valid_negatives == min_valid_hard_negative][0]])

    if not anchors or not positives or not semi_positives or not negatives:
        print("Warning: No valid quartets found!")
        return None, None, None, None

    return torch.stack(anchors).to(device), torch.stack(positives).to(device), torch.stack(semi_positives).to(device), torch.stack(negatives).to(device)

def sample_triplets(latent_rep, y_batch):
    """Sample semi-hard triplets."""
    n = latent_rep.size(0)
    distances = pairwise_distances(latent_rep)

    positive_mask = y_batch.view(n, 1) == y_batch.view(1, n)
    negative_mask = ~positive_mask

    anchors, positives, negatives = [], [], []
    for i in range(n):
        valid_positives = distances[i, positive_mask[i]]
        valid_negatives = distances[i, negative_mask[i]]

        if len(valid_positives) == 0 or len(valid_negatives) == 0:
            continue

        max_positive_dist = valid_positives.max()

        semi_hard_negatives = valid_negatives[valid_negatives < max_positive_dist]

        if len(semi_hard_negatives) == 0:
            continue

        min_semi_hard_negative = semi_hard_negatives.min()

        anchors.append(latent_rep[i])
        positives.append(
            latent_rep[torch.nonzero(positive_mask[i], as_tuple=True)[0][valid_positives == max_positive_dist][0]])
        negatives.append(
            latent_rep[torch.nonzero(negative_mask[i], as_tuple=True)[0][valid_negatives == min_semi_hard_negative][0]])

    if not anchors or not positives or not negatives:
        print("Warning: No valid triplets found!")
        return None, None, None

    return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)

def compute_quartet_loss_function(anchor, positive, semi_positive, negative, margin1=0.2, margin2=0.5):
    """
    Compute the quartet loss
    """
    distance_positive = (anchor - positive).pow(2).sum(1)
    distance_semi_positive = (anchor - semi_positive).pow(2).sum(1)
    distance_negative = (anchor - negative).pow(2).sum(1)
    losses = F.relu(distance_positive - distance_semi_positive + margin1) + \
             F.relu(distance_semi_positive - distance_negative + margin2)
    return losses.mean()

def compute_triplet_loss_function(anchor, positive, negative, margin=0.2):
    """
    Compute the triplet loss
    """
    distance_positive = (anchor - positive).pow(2).sum(1)
    distance_negative = (anchor - negative).pow(2).sum(1)
    losses = F.relu(distance_positive - distance_negative + margin)
    return losses.mean()

def calculate_quartet_loss_from_graph(graph, model, device, le, family_map, family_only=False):
    latent_reps = torch.stack([data['attr_dict']['latent'] for _, data in graph.nodes(data=True)]).to(device)
    labels = torch.tensor([data['attr_dict']['true_label'] for _, data in graph.nodes(data=True)], dtype=torch.long).to(device)

    if family_only:
        anchor, positive, negative = sample_triplets(latent_reps, labels)
        if anchor is not None and positive is not None and negative is not None:
            loss = compute_triplet_loss_function(anchor, positive, negative)
            return loss
        else:
            print("Warning: No valid triplets found. Returning zero loss.")
            return torch.tensor(0.0).to(device)

    anchor, positive, semi_positive, negative = sample_quartets(latent_reps, labels, le, family_map, device)
    if anchor is not None and positive is not None and semi_positive is not None and negative is not None:
        loss = compute_quartet_loss_function(anchor, positive, semi_positive, negative)
        return loss

    print("Warning: No valid quartets found. Reverting to triplet loss.")
    anchor, positive, negative = sample_triplets(latent_reps, labels)
    if anchor is not None and positive is not None and negative is not None:
        loss = compute_triplet_loss_function(anchor, positive, negative)
        return loss

    print("Warning: No valid triplets found. Returning zero loss.")
    return torch.tensor(0.0).to(device)
