from cuml.decomposition import PCA as cuPCA
from cuml.decomposition import IncrementalPCA as cuIPCA
import cudf
import numpy as np
import datetime as datetime
import pandas as pd
import pickle


# Parameters
#input_file = 'msmarco1000_embeddings.npy'
#output_file = 'msmarco1000_embeddings_reduced.npy'
input_file = 'msmarco_embeddings.npy'
output_file = 'msmarco_embeddings_reduced.npy'
n_components = 192  # Target number of dimensions

# Load the entire matrix and convert to cuDF DataFrame for GPU processing
matrix_np = np.load(input_file)
print("Loaded matrix")

# initialize the incremental PCA with cuml
ipca = cuIPCA(n_components=n_components, batch_size=1024)


# Convert to cuDF, each time only 1024 rows are converted
for i in range(0, len(matrix_np), 1024):
    now = datetime.datetime.now()
    current_time = now.strftime("%H:%M:%S")
    up = min(i + 1024, len(matrix_np))
    print(current_time, "Converting %d to %d" % (i, up))
    matrix_df = cudf.DataFrame(matrix_np[i:up])
    # fit the incremental PCA
    ipca.partial_fit(matrix_df)

print("Fitted matrix")

# Save the fitted PCA model
with open('ipca_msmarco.pkl', 'wb') as f:
    pickle.dump(ipca, f)

exit()

# now transform the entire matrix with batch size 1024 based on the fitted PCA
transformed_matrix = np.zeros((0, n_components), dtype=np.float32)
for i in range(0, len(matrix_np), 1024):
    now = datetime.datetime.now()
    current_time = now.strftime("%H:%M:%S")
    up = min(i + 1024, len(matrix_np))
    print(current_time, "Transforming %d to %d" % (i, up))
    matrix_df = cudf.DataFrame(matrix_np[i:up])
    # transform the matrix
    reduced_matrix_df = ipca.transform(matrix_df)   
    # concatenate the transformed matrix
    reduced_matrix_np = reduced_matrix_df.to_pandas().values
    transformed_matrix = np.concatenate((transformed_matrix, reduced_matrix_np), axis=0)

print("Transformed matrix")
print(transformed_matrix.shape)  # Should be (number of documents, 200)
np.save(output_file, transformed_matrix)
