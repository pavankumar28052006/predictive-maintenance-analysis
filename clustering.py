import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import config

CLUSTER_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

def run_clustering(X):
    data = X[CLUSTER_COLUMNS].copy()
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    scores = {}
    print("\nK-Means Clustering")

    for k in range(2, 8):
        model = KMeans(
            n_clusters=k,
            random_state=config.RANDOM_STATE,
            n_init=10
        )
        labels = model.fit_predict(scaled_data)
        score = silhouette_score(scaled_data, labels)
        scores[k] = score
        print(f"K={k}, Silhouette Score={score:.4f}")

    best_k = max(scores, key=scores.get)
    print("\nBest number of clusters:", best_k)

    kmeans = KMeans(
        n_clusters=best_k,
        random_state=config.RANDOM_STATE,
        n_init=10
    )
    data["cluster"] = kmeans.fit_predict(scaled_data)

    save_cluster_summary(data)
    plot_silhouette_scores(scores)
    plot_clusters(data)

    return kmeans, scaler

def save_cluster_summary(data):
    summary = data.groupby("cluster").mean(numeric_only=True)
    summary.to_csv(os.path.join(config.OUTPUT_DIR, "cluster_summary.csv"))

def plot_silhouette_scores(scores):
    plt.figure(figsize=(8, 5))
    plt.plot(list(scores.keys()), list(scores.values()), marker="o")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score for Different K Values")
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, "silhouette_scores.png"))
    plt.close()

def plot_clusters(data):
    plt.figure(figsize=(9, 6))
    scatter = plt.scatter(
        data["Rotational speed [rpm]"],
        data["Torque [Nm]"],
        c=data["cluster"],
        alpha=0.6
    )
    plt.xlabel("Rotational Speed (rpm)")
    plt.ylabel("Torque (Nm)")
    plt.title("Machine Operating State Clusters")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, "machine_clusters.png"))
    plt.close()