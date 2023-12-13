import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pprint import pprint
from sklearn.utils import shuffle
import os

"""
Do not include benign/normal class in family map.
"""

def data_loader(classes, dataset_name, family=False):
    """
    Load data based on the given classes and dataset name.

    Parameters:
    - classes (str or list): Class names or "random".
    - dataset_name (str): Name of the dataset ("kdd" or "cicids").

    Returns:
    - Data based on the dataset function.
    """
    if dataset_name == "kdd":
        data, family_map = kdd_dataloader()
    elif dataset_name == "cicids":
        data, family_map = cicids_dataloader()
    elif dataset_name == "nsl-kdd":
        data, family_map = nsl_kdd_dataloader(family)
    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")

    X = data.drop(columns=['label'])
    y = data['label']

    X = pd.get_dummies(X)

    y_encoded, le, unhidden_count = rearrange_encoder(y, classes)


    (unhidden_X, unhidden_y), (hidden_X, hidden_y) = create_datasets(y_encoded, X, unhidden_count)

    scaler = StandardScaler()
    unhidden_X = scaler.fit_transform(unhidden_X)
    hidden_X = scaler.transform(hidden_X)

    X_train, X_test, y_train, y_test = train_test_split(unhidden_X, unhidden_y, test_size=0.5, stratify=unhidden_y)

    return X_train, y_train, X_test, y_test, hidden_X, hidden_y, le, unhidden_count, family_map


def nsl_kdd_dataloader(family=False):
    """
    Load NSL-KDD dataset.

    Returns:
    - Data based on the NSL-KDD dataset.
    - Family mapping for the NSL-KDD dataset.
    """
    columns = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment",
               "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted",
               "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
               "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
               "srv_rerror_rate",
               "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
               "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
               "dst_host_srv_diff_host_rate",
               "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
               "attack_type", "difficulty_level"]

    script_dir = os.path.dirname(os.path.abspath(__file__))

    train_data_path = os.path.join(script_dir, '../../data/NSL-KDD/KDDTrain+.txt')
    test_data_path = os.path.join(script_dir, '../../data/NSL-KDD/KDDTest+.txt')

    train_data = pd.read_csv(train_data_path, header=None, names=columns)
    test_data = pd.read_csv(test_data_path, header=None, names=columns)

    data = pd.concat([train_data, test_data], axis=0)

    data = data.drop(columns=['difficulty_level'])

    data.rename(columns={'attack_type': 'label'}, inplace=True)

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
    if family:
        data['label'] = data['label'].map(family_map).fillna(data['label'])

    return data, family_map

def kdd_dataloader():
    """
    Load KDD dataset based on the given classes.

    Parameters:
    - classes (str or list): Class names or "random".

    Returns:
    - Data and family mapping based on the KDD dataset.
    """
    data = pd.read_csv('data/kddcup99.csv')

    family_map = {
        "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS", "smurf": "DoS", "teardrop": "DoS",
        "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L", "multihop": "R2L", "phf": "R2L", "spy": "R2L",
        "warezclient": "R2L", "warezmaster": "R2L",
        "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R", "rootkit": "U2R",
        "ipsweep": "Probing", "nmap": "Probing", "portsweep": "Probing",
        "satan": "Probing"
    }

    return data, family_map

def cicids_dataloader():
    """
    Load CICIDS dataset based on the given classes.

    Parameters:
    - classes (str or list): Class names or "random".

    Returns:
    - Data and family mapping based on the CICIDS dataset.
    """
    data = pd.read_csv('data/CIC-IDS2017-undersampled.csv')

    data.columns = [col.strip() for col in data.columns]
    data.rename(columns={'Label': 'label'}, inplace=True)

    family_map = {
        "DoS Hulk": "DoS",
        "DoS GoldenEye": "DoS",
        "DoS slowloris": "DoS",
        "DoS Slowhttptest": "DoS",
        "DDoS": "DoS",
        "PortScan": "PortScan",
        "FTP-Patator": "Patator",
        "SSH-Patator": "Patator",
        "Bot": "Exploits",
        "Web Attack – Brute Force": "Web Attack",
        "Web Attack – XSS": "Web Attack",
        "Web Attack – Sql Injection": "Web Attack",
        "Infiltration": "Exploits",
        "Heartbleed": "Exploits"
    }

    return data, family_map

def rearrange_encoder(y, hidden_classes):
    """
    Rearrange the encoder to place the hidden classes at the end.

    Parameters:
    - y (Series): Target variable with categorical labels.
    - hidden_classes (str or list): Hidden class or classes.

    Returns:
    - y_encoded (array): Encoded target variable.
    - le (LabelEncoder): Modified label encoder.
    - unhidden_count (int): Count of unhidden classes.
    """
    if not isinstance(hidden_classes, list):
        hidden_classes = [hidden_classes]

    le = preprocessing.LabelEncoder()
    le.fit(y.unique())

    hidden_indices = [np.where(le.classes_ == cls)[0][0] for cls in hidden_classes]

    le.classes_ = np.delete(le.classes_, hidden_indices)

    for hidden_class in hidden_classes:
        le.classes_ = np.append(le.classes_, hidden_class)

    y_encoded = le.transform(y)

    unhidden_count = len(le.classes_) - len(hidden_classes)

    return y_encoded, le, unhidden_count

def print_label_encoder(le, unhidden_count):
    """
    Print the label encoder in a special format.

    Parameters:
    - le (LabelEncoder): The label encoder.
    - unhidden_count (int): Count of unhidden classes.
    """
    classes = le.classes_
    hidden_count = len(classes) - unhidden_count

    for i in range(hidden_count, 0, -1):
        print(f"{classes[-i]}: {-i}")

    for i in range(unhidden_count):
        print(f"{classes[i]}: {i}")

    print("\nNormal Format:")
    for i, cls in enumerate(classes):
        print(f"{cls}: {i}")

def create_datasets(y_encoded, X, unhidden_count):
    hidden_indices = np.where(y_encoded >= unhidden_count)[0]
    unhidden_indices = np.where(y_encoded < unhidden_count)[0]

    unhidden_samples = []
    for i in range(unhidden_count):
        class_indices = np.where(y_encoded[unhidden_indices] == i)[0]
        selected_indices = shuffle(class_indices)[:min(500, len(class_indices))]
        unhidden_samples.extend(unhidden_indices[selected_indices])

    unhidden_X = X.iloc[unhidden_samples]

    leftover_unhidden_indices = list(set(unhidden_indices) - set(unhidden_samples))
    leftover_unhidden_dataset = X.iloc[leftover_unhidden_indices]

    hidden_samples = []
    for i in range(unhidden_count, len(np.unique(y_encoded))):
        class_indices = np.where(y_encoded[hidden_indices] == i)[0]
        selected_indices = shuffle(class_indices)[:min(500, len(class_indices))]
        hidden_samples.extend(hidden_indices[selected_indices])

    additional_samples = shuffle(leftover_unhidden_indices)[:min(500, len(leftover_unhidden_indices))]
    hidden_samples.extend(additional_samples)

    hidden_X = X.iloc[hidden_samples]

    unhidden_y = y_encoded[unhidden_samples]
    hidden_y = y_encoded[hidden_samples]

    return (unhidden_X, unhidden_y), (hidden_X, hidden_y)
