import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
from src.models.autoencoder import Autoencoder  # Replace with actual import
from src.datasets.dataloader import data_loader
import numpy as np
from tqdm import tqdm

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

def baseline(hidden_class):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train, y_train, _, _, X_hidden, y_hidden, le, unhidden_count, _ = data_loader([hidden_class], "nsl-kdd", False)
    input_dim = X_train.shape[1]

    X_train_tensor = torch.Tensor(X_train).to(device)
    X_hidden_tensor = torch.Tensor(X_hidden).to(device)

    baseline_model = Autoencoder(input_dim=input_dim, encoding_dim=128, num_classes=4).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(baseline_model.parameters(), lr=0.00001)

    train_dataset = TensorDataset(X_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)

    num_epochs = 400
    for epoch in range(num_epochs):
        running_loss = 0.0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}",
                            leave=False)
        for batch_idx, (data,) in progress_bar:
            data = data.to(device)
            optimizer.zero_grad()
            decoded, _, _ = baseline_model(data)  # Call once and unpack the tuple
            loss = criterion(decoded, data)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_description(f"Epoch {epoch + 1}/{num_epochs} - Loss: {running_loss / (batch_idx + 1):.4f}")
        tqdm.write(f"Epoch {epoch + 1} - Average Loss: {running_loss / len(train_loader):.4f}")

    z_score_multiplier = 3

    baseline_model.eval()
    hidden_loader = DataLoader(TensorDataset(X_hidden_tensor), batch_size=100, shuffle=False)
    reconstruction_errors = []

    with torch.no_grad():
        for batch_idx, (data,) in enumerate(hidden_loader):
            data = data.to(device)
            decoded, _, _ = baseline_model(data)
            reconstruction_error = torch.mean((data - decoded).pow(2), dim=1)
            reconstruction_errors.append(reconstruction_error)

    reconstruction_errors = torch.cat(reconstruction_errors)
    mean_error = torch.mean(reconstruction_errors).item()
    std_error = torch.std(reconstruction_errors).item()
    threshold = mean_error + (z_score_multiplier * std_error)
    print(f"Reconstruction Threshold: {threshold}")

    anomalies = reconstruction_errors > threshold

    hidden_classes_indices = list(range(unhidden_count, len(le.classes_)))
    hidden_samples_count = 0

    for i, is_anomaly in enumerate(anomalies):
        if y_hidden[i] in hidden_classes_indices:
            hidden_samples_count += 1
            print(
                f"Sample {i} of hidden class with label {y_hidden[i]} has reconstruction error: {reconstruction_errors[i].item()}")

            if is_anomaly:
                print(f"This sample is detected as an anomaly!")

    print(f"\nTotal number of samples from hidden classes: {hidden_samples_count}")

    print("\nClass counts in hidden dataset:")
    unique_classes, counts = np.unique(y_hidden, return_counts=True)
    for unique_class, count in zip(unique_classes, counts):
        if unique_class < unhidden_count:
            print(f"({unique_class}) {le.classes_[unique_class]}: {count}")
        else:
            adjusted_index = unique_class - len(le.classes_)
            print(f"({adjusted_index}) {le.classes_[unique_class]}: {count}")

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for i, is_anomaly in enumerate(anomalies):
        if y_hidden[i] in hidden_classes_indices:  # If it's a hidden class sample
            if is_anomaly:
                TP += 1  # True Positive
            else:
                FN += 1  # False Negative
        else:  # If it's not a hidden class sample
            if is_anomaly:
                FP += 1  # False Positive
            else:
                TN += 1  # True Negative

    print(f"Total dataset size: {len(y_hidden)}")
    print(f"True Positives: {TP}")
    print(f"True Negatives: {TN}")
    print(f"False Positives: {FP}")
    print(f"False Negatives: {FN}")

    if TP + FP == 0:
        precision = 0
    else:
        precision = TP / (TP + FP)

    if TP + FN == 0:
        recall = 0
    else:
        recall = TP / (TP + FN)

    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    print(f"F1 Score: {f1:.4f}")

    return TP, TN, FP, FN, f1

if __name__ == '__main__':
    total_TP = 0
    total_TN = 0
    total_FP = 0
    total_FN = 0
    total_f1 = 0

    for attack_class in family_map.keys():
        print(f"Evaluating for hidden class: {attack_class}")
        TP, TN, FP, FN, f1 = baseline(attack_class)
        total_TP += TP
        total_TN += TN
        total_FP += FP
        total_FN += FN
        total_f1 += f1

    avg_f1 = total_f1 / len(family_map)

    print("\nFinal Evaluation Metrics:")
    print(f"Total True Positives: {total_TP}")
    print(f"Total True Negatives: {total_TN}")
    print(f"Total False Positives: {total_FP}")
    print(f"Total False Negatives: {total_FN}")
    print(f"Average F1 Score: {avg_f1:.4f}")
