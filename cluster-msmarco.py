import numpy as np
from cuml.cluster import KMeans
import cudf

# Load the dataset
#input_file = 'msmarco_embeddings_reduced.npy'
input_file = 'msmarco_embeddings_reduced.npy'
#input_file = 'msmarco1000_embeddings_reduced.npy'
matrix = np.load(input_file)

# Number of clusters
#n_clusters = 4 * 32 * 10
n_clusters = 100

# Convert the numpy array to cuDF DataFrame for GPU processing
matrix_df = cudf.DataFrame(matrix)
print("Loaded matrix")

# Perform K-means clustering
kmeans = KMeans(n_clusters=n_clusters)
print("Fitting k-means model")
kmeans.fit(matrix_df)
print("Fitted k-means model")

# Extract cluster labels
labels = kmeans.labels_.to_numpy()
#print("Extracted labels")

# save the labels to a file
prefix = 'msmarco-' + str(n_clusters) + '-cluster'
np.save(prefix + '-labels.npy', labels)

# Sort indices by labels to group them
sorted_indices = np.argsort(labels)
sorted_labels = labels[sorted_indices]
#print("Sorted labels", sorted_labels)
#print("Sorted indices", sorted_indices)

# Calculate distances from each point to its centroid
centroids = kmeans.cluster_centers_.to_numpy()  # Transfer centroids to CPU memory
np.save(prefix + '-centroids.npy', centroids)

sorted_matrix = matrix[sorted_indices]
distances = np.linalg.norm(sorted_matrix - centroids[sorted_labels], axis=1)

# Find the index of the minimum distance in each cluster
size_of_each_cluster = np.bincount(sorted_labels)
print("size of each cluster", size_of_each_cluster)

representative_indices = np.zeros(n_clusters, dtype=int)
start_idx = 0
for i in range(n_clusters):
    # Find the end of the current cluster
    end_idx = start_idx
    while end_idx < len(labels) and sorted_labels[start_idx] == sorted_labels[end_idx]:
        end_idx += 1
    
    # Find the index of the minimum distance in this cluster
    min_dist_idx = start_idx + np.argmin(distances[start_idx:end_idx])
    representative_indices[i] = sorted_indices[min_dist_idx]
    
    # Move to the next cluster
    start_idx = end_idx

# representative_indices now contains the indices of the representative vectors
#print("Indices of representative vectors:", representative_indices)

# Save the indices to a file
np.save(prefix + '-reps.npy', representative_indices)

