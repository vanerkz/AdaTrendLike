import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os

def save_dataset_inputs(true_x_np, rec_dnp, mu_trendnp, cov_trendnp, test_y_label,pi_hat_np, dataset_name, save_dir='temp_inputs'):
    """
    Save input arrays for a dataset as .npy files.

    Parameters:
    - true_x_np: np.array, ground truth values
    - rec_dnp: np.array, reconstructed values from model
    - mu_trendnp: np.array, predicted trend mean
    - cov_trendnp: np.array, predicted trend covariance
    - test_y_label: np.array, 0/1 labels for normal/anomaly
    - dataset_name: str, name of the dataset
    - save_dir: str, folder to save files
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    np.save(os.path.join(save_dir, f'{dataset_name}_true_x.npy'), true_x_np)
    np.save(os.path.join(save_dir, f'{dataset_name}_rec_d.npy'), rec_dnp)
    np.save(os.path.join(save_dir, f'{dataset_name}_mu_trend.npy'), mu_trendnp)
    np.save(os.path.join(save_dir, f'{dataset_name}_cov_trend.npy'), cov_trendnp)
    np.save(os.path.join(save_dir, f'{dataset_name}_labels.npy'), test_y_label)
    np.save(os.path.join(save_dir, f'{dataset_name}_pi_hat.npy'), pi_hat_np)

    print(f"Saved all input arrays for dataset '{dataset_name}' in '{save_dir}'")

def k_means_clustering(A_score_np):

    kmeans = KMeans(n_clusters=2, random_state=0).fit(A_score_np.reshape(-1,1))
    labels = kmeans.labels_

    cluster0 = A_score_np[labels == 0]
    cluster1 = A_score_np[labels == 1]

    # compute means
    mean0 = cluster0.mean()
    mean1 = cluster1.mean()

    # select normal cluster (smaller mean)
    if mean0 > mean1:
        larger_cluster = cluster0
    else:
        larger_cluster = cluster1

    # threshold from normal cluster
    threshold = larger_cluster.mean()
    std = larger_cluster.std()
    return threshold, std

import numpy as np
from sklearn.cluster import KMeans

def kmeans(A_score_np):
    # reshape for KMeans
    X = A_score_np.reshape(-1, 1)
    
    # fit 2 clusters
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    labels = kmeans.labels_

    # separate clusters
    cluster0 = A_score_np[labels == 0]
    cluster1 = A_score_np[labels == 1]

    # select larger cluster by number of points
    if len(cluster0) > len(cluster1):
        larger_cluster = cluster0
    else:
        larger_cluster = cluster1

    # std of the larger cluster
    larger_cluster_std = larger_cluster.std()


    return larger_cluster_std

def noise_inject(train, val,test, labels, anomaly_ratio=0.1, random_seed=None):
    """
    Add anomalies from test to training data to reach target anomaly ratio.
    
    Parameters:
    - train: np.array, training data
    - test: np.array, test data
    - labels: np.array, test labels (1=anomaly, 0=normal)
    - anomaly_ratio: float, target ratio of anomalies in updated training set
    - random_seed: int, for reproducibility
    
    Returns:
    - train_scaled: np.array, updated training data
    - test_scaled: np.array, updated test data
    - labels_new: np.array, updated test labels
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Count existing anomalies in train
    # If your train doesn't have labels, assume 0 anomalies
    num_train_anomalies = 0  # adjust if train has anomaly info

    # Indices of anomalies in test
    anomaly_indices = np.where(labels == 1)[0]

    # Calculate number of anomalies needed to reach target ratio
    total_train_needed = int((len(train) + len(anomaly_indices)) * anomaly_ratio)  # optional upper bound
    num_to_add = max(0, int((anomaly_ratio * (len(train) + num_train_anomalies)) - num_train_anomalies))
    num_to_add = min(num_to_add, len(anomaly_indices))

    if num_to_add == 0:
        print("No anomalies added (already at or above target ratio).")
        train_new = train
        test_new = test
        labels_new = labels
    else:
        # Randomly select anomalies
        selected_indices = np.random.choice(anomaly_indices, size=num_to_add, replace=False)
        anomalies_to_add = test[selected_indices]

        # Add to training data
        train_new = np.concatenate((train, anomalies_to_add), axis=0)

        # Remove these anomalies from test and labels
        mask = np.ones(len(test), dtype=bool)
        mask[selected_indices] = False
        test_new = test[mask]
        labels_new = labels[mask]

        print(f"Added {num_to_add} anomalies. New train anomaly ratio: {num_to_add / len(train_new):.3f}")

    # Scale
    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(train_new)
    train_scaled = scaler.transform(train_new)
    val_scaled = scaler.transform(val)
    test_scaled = scaler.transform(test_new)

    return train_scaled, val_scaled,test_scaled, labels_new
