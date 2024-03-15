import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime
import heapq
import os

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

# Load the graph data
graph_file = "graph_permuted.npy" 
#graph_file = "015graph.npy"
graph = np.load(graph_file)

print("Graph shape: ", graph.shape)

# load the PCA model
with open('ipca_msmarco.pkl', 'rb') as f:
    ipca = pickle.load(f)

# check if the number of components in the PCA model is the same as the number of dimensions in the embeddings
if embeddings.shape[1] == ipca.n_components:
    # transform the query embeddings
    print("Transforming query embeddings with PCA model...")
    #query_embeddings = ipca.transform(query_embeddings)
    DIM_REDUCED = True
else:
    DIM_REDUCED = False
    print("No need to transform")

def generate_query_embeddings(sentences):
    tmp = model.encode(sentences)
    if DIM_REDUCED:
        #print("Transforming query embeddings with PCA model...")
        tmp = ipca.transform(tmp)
    return tmp
    #return model.encode(sentences)

def find_approximate_nearest_neighbors(query_vector, k = 100, step = 10, hidden_id = None):
    hit_step = 99999

    # first find the representative vector that is closest to the query vector, based on dot product similarity
    distances = np.linalg.norm(rep_vector - query_vector, axis=1) #this is based on L2
    #print("distance to the rep vectors: ", distances)
    entry_distance = np.min(distances)
    entry_idx = reps[np.argmin(distances)]

    visited = set([entry_idx])

    # the to_be_explored list is a priority queue containing tuples
    # each tuples contains the distance to the query_vector and the index

    to_be_explored = []
    heapq.heappush(to_be_explored, (entry_distance, entry_idx))

    for i in range(step):
        #print("Step: ", i)
        #print("to_be_explored: ", to_be_explored)
        #print("graph[to_be_explored]: ", graph[to_be_explored])
        _, current_idx = heapq.heappop(to_be_explored)
        for j in graph[current_idx]:
            if j not in visited:
                visited.add(j)
                distance = np.linalg.norm(query_vector - embeddings[j])
                heapq.heappush(to_be_explored, (distance, j))
                if hidden_id != None and j == hidden_id:
                    hit_step = i
    
    # now find the top k nearest neighbors
    # we insert all the explored idx and their distances into a list
    visited_tuples = []
    for i in visited:
        distance = np.linalg.norm(query_vector - embeddings[i])
        visited_tuples.append((distance, i))
    
    # sort the list based on the distance
    visited_tuples.sort(key = lambda x: x[0])

    # now return the top k nearest neighbors, their distances and the step when the hidden_id is found
    distances = [x[0] for x in visited_tuples[:k]]
    ids = [x[1] for x in visited_tuples[:k]]

    return distances, ids, hit_step

def query_index(sentence, k = 10):
    if isinstance(sentence, str):
        sentence = [sentence]
    embeddings = generate_query_embeddings(sentence)
    result_distance = []
    result_idx = []
    for i in range(len(embeddings)):
        distances, original_idx, _ = find_approximate_nearest_neighbors(embeddings[i], k)
        result_distance.append(distances)
        result_idx.append(original_idx)
    return result_distance, result_idx


# try the first 100 vectors

hit_step_history = []

for i in range(1000):
    #print("Query: ", i)
    distances, indices, step = find_approximate_nearest_neighbors(embeddings[i], k=10, step=15, hidden_id=i)
    hit_step_history.append(step)
    #print("Step: ", step)
    #print("Distances: ", distances)
    #print("Indices: ", indices)

histogram = np.bincount(hit_step_history)
histogram = histogram / np.sum(histogram)

print("Average step: ", sum(hit_step_history)/len(hit_step_history))
print("Histogram: ", histogram[0:20])
print("Prob of the first 10: ", sum(histogram[0:10]))

def read_queries(filepath):
    """
    Read queries from a file where each row contains a QuestionID and a sentence,
    separated by a tab.
    """
    queries = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            question_id, sentence = line.strip().split('\t')
            queries.append((question_id, sentence))
    return queries

def output_results_to_file(query_embeddings, doc_ids, output_filepath, k=10, step = 20):
    """
    Process each query, generate embeddings, make the query and output the results to a file.
    """

    # measure the time to process the queries
    start = time.time()
    current_time = datetime.now().strftime("%H:%M:%S")
    print(current_time + ": Processing queries...")
    with open(output_filepath, 'w', encoding='utf-8') as outfile:
        t = 0
        for question_id, sentence in queries:
            #query_embedding = model.encode([sentence])  # Note: [sentence] to keep input as batch
            #distances, indices = query_index(sentence, k)
            distances, indices, _ = find_approximate_nearest_neighbors(query_embeddings[t], k, step = step, hidden_id = None)
            outfile.write(f"Query: {question_id} {sentence}\n")
            for i in range(len(indices)):    
                doc_id = doc_ids[indices[i]]
                distance = distances[i]
                outfile.write(f"{doc_id}\n")
            outfile.write("----------\n\n")
            t += 1
            if t % 100 == 0:
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                print(current_time + ":Processed", t,  " queries")
    end = time.time()
    current_time = datetime.now().strftime("%H:%M:%S")
    print(current_time + ":Processed", t,  " queries")

    # measure the time to process the queries in seconds
    print("Time to process the queries: ", end - start)
    print("Average time to process each query: ", (end - start)/query_embeddings.shape[0])


query_filepath = 'msmarco-queries-1000.tsv'
result_filepath = 'graph-msmarco-results.txt'
query_embedding_file = 'msmarco-queries-1000-embeddings.npy'

queries = read_queries(query_filepath)

# test if the query embedding file exists
if os.path.exists(query_embedding_file):
    query_embeddings = np.load(query_embedding_file)
else :
    print("Processing", len(queries), "queries")
    start = time.time()
    query_embeddings = generate_query_embeddings([sentence for question_id, sentence in queries])
    end = time.time()
    print("Time to generate query embeddings: ", end - start)
    print("Average time to generate each query embedding: ", (end - start)/len(queries))
    print(query_embeddings.shape)
    np.save(query_embedding_file, query_embeddings) 

output_results_to_file(query_embeddings, doc_ids, result_filepath, k = 100, step = 20)
