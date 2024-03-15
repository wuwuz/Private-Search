import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime
import heapq
import os
import requests

from cuml.decomposition import PCA as cuPCA
from cuml.decomposition import IncrementalPCA as cuIPCA
import cudf
import pickle
import time

OUTPUT_FILE = "cluster-msmarco-results.txt"
CLUSTER_NUM = 4 * 32 * 10
#CLUSTER_NUM = 100
DIM_REDUCED = False

# Initialize your model for embeddings
model = SentenceTransformer('msmarco-distilbert-base-tas-b')

# Load the embeddings data
#embeddings_file = "msmarco_embeddings_reduced.npy"
embeddings_file = "msmarco_embeddings_reduced_permuted.npy"
embeddings = np.load(embeddings_file)

#docid_file = "msmarco_docid.npy"
docid_file = "msmarco_docid_permuted.npy"
doc_ids = np.load(docid_file)

# Load the cluster representatives
representatives_file = "msmarco-" + str(CLUSTER_NUM) + "-cluster-reps-relabeled.npy"
reps = np.load(representatives_file)

# now extract the representative vectors for each cluster, based on the representative indices
rep_vector = embeddings[reps]
np.save("msmarco-" + str(CLUSTER_NUM) + "-reps-vector.npy", rep_vector)

# Load the graph data
graph_file = "graph_permuted.npy"
#graph_file = "015graph.npy"
graph = np.load(graph_file)
rep_graph = graph[reps]
np.save("msmarco-" + str(CLUSTER_NUM) + "-reps-neighbors.npy", rep_graph)