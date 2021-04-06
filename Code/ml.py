import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import TruncatedSVD

def matrix_track(tracker_df):
    matrix_dict = dict()

    min_x = min(tracker_df["altered_x"])
    max_x = max(tracker_df["altered_x"])
    min_y = min(tracker_df["altered_y"])
    max_y = max(tracker_df["altered_y"])

    grouped = tracker_df.groupby("UniqueID")

    x_dim = max_x - min_x
    y_dim = max_y - min_y

    for name, group in grouped:
        image_array = np.zeros((x_dim, y_dim), dtype=int)
        for _, row in group.iterrows():
            if row["altered_x"] > 1920 or row["altered_y"] > 1080:
                continue
            else:
                image_array[(row["altered_x"]-1) - min_x, (row["altered_y"]-1) - min_y] = 1
        matrix_dict[name] = image_array
    return matrix_dict

def matrix_to_vector(matrix_dict):
    for key, value in matrix_dict.items():
        length, height = value.shape
        matrix_dict[key] = value.reshape((length * height, 1))
    return matrix_dict

def train_vectors(vector):
    uniqueid = []
    vectors = []

    for key, value in vector.items():
        uniqueid.append(key)
        vectors.append(value)

    vectors = np.array(vectors)
    nsamples, nx, ny = vectors.shape
    vectors = vectors.reshape((nsamples,nx*ny))
    return uniqueid, vectors

def reduce_dimensions(matrix, n_components=200):
    tsvd = TruncatedSVD(n_components)
    X_sparse_tsvd = tsvd.fit(matrix).transform(matrix)
    return X_sparse_tsvd

def calculate_best_n_clusters(vectors, range_n_clusters):
    score, n, labels = [], [], []
    for n_clusters in range_n_clusters:
        k_model = k_mean_model(vectors, n_clusters)
        cluster_labels = k_predict(k_model, vectors)
        labels.append(cluster_labels)

        silhouette_avg = silhouette_score(vectors, cluster_labels)
        score.append(silhouette_avg)
        n.append(n_clusters)
        print(f"For n_clusters = {n_clusters} The average silhouette_score is : {silhouette_avg}")

    max_value = max(score)
    max_index = score.index(max_value) 
    return n[max_index], labels[max_index], k_model

def k_mean_model(vectors, n_clusters):
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    k_mean_model = clusterer.fit(vectors)
    return k_mean_model

def k_predict(k_model, vectors):
    cluster_labels = k_model.predict(vectors)
    return cluster_labels

def run_all(tracker_df, range_n_clusters = [2,3,4,5,6,7,8,9,10,11,12,13,14,15]):
    matrix = matrix_track(tracker_df)
    vector = matrix_to_vector(matrix)
    uniqueid, vectors = train_vectors(vector)
    n_clusters, labels, model = calculate_best_n_clusters(vectors, range_n_clusters)
    return n_clusters, labels, uniqueid, model

if __name__ == "__main__":
    n_clusters, labels, uniqueid, model = run_all(g6.tracker_df, [6])