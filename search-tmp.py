import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime

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
embeddings_file = "msmarco_embeddings_reduced.npy"
embeddings = np.load(embeddings_file)

#docid_file = "msmarco_docid.npy"
docid_file = "msmarco_docid.npy"
doc_ids = np.load(docid_file)

# Load the cluster representatives
representatives_file = "msmarco-" + str(CLUSTER_NUM) + "-cluster-reps.npy"
reps = np.load(representatives_file)

# Load the cluster labels for each embedding vector
labels_file = "msmarco-" + str(CLUSTER_NUM) + "-cluster-labels.npy"
labels = np.load(labels_file)
#print("labels: ", labels)

# sort the vectors, so that the vectors in the same cluster are grouped together
sorted_indices = np.argsort(labels)
#sorted_labels = labels[sorted_indices]
sorted_embeddings = embeddings[sorted_indices]

# now decide the start and end of each cluster
num_vector_in_each_cluster = np.bincount(labels)
# now build a prefix sum use pure numpy operation
start_idx = np.cumsum(num_vector_in_each_cluster)
start_idx = np.concatenate(([0], start_idx))


# now extract the representative vectors for each cluster, based on the representative indices
rep_vector = embeddings[reps]

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

def find_approximate_nearest_neighbors(query_vector, k = 100, tiptoe = False):
    # first find the representative vector that is closest to the query vector, based on dot product similarity
    distances = np.linalg.norm(rep_vector - query_vector, axis=1) #this is based on L2
    #print("distance to the rep vectors: ", distances)
    cluster_idx = np.argmin(distances)
    #print("this one is the cloeset: ", cluster_idx)
    # now find the top k nearest vectors in the cluster
    start, end = start_idx[cluster_idx], start_idx[cluster_idx + 1]
    #distance_to_cluster = np.linalg.norm(sorted_embeddings[start:end] - query_vector, axis=1)

    if tiptoe == True:
        # compute the dot product to the vectors in the cluster
        distance_to_cluster = np.dot(sorted_embeddings, query_vector)[start:end]
    else:
        # compute the dot product to the vectors in the cluster
        distance_to_cluster = np.dot(sorted_embeddings[start:end], query_vector)
    # find the top k nearest vectors with largest dot product
    # sort them in decending order based on the dot product
    top_k_idx = np.argsort(distance_to_cluster)[::-1][:k]
    #print("top k index", top_k_idx)
    original_idx = sorted_indices[start:end][top_k_idx]
    distances = distance_to_cluster[top_k_idx]
    return distances, original_idx

def query_index(sentence, k = 10):
    if isinstance(sentence, str):
        sentence = [sentence]
    embeddings = generate_query_embeddings(sentence)
    result_distance = []
    result_idx = []
    for i in range(len(embeddings)):
        distances, original_idx = find_approximate_nearest_neighbors(embeddings[i], k)
        result_distance.append(distances)
        result_idx.append(original_idx)
    return result_distance, result_idx

#Example sentences to query
example_sentences = ["Physics is cool", "I don't know what the hell"]
#query_embeddings = generate_query_embeddings(example_sentences)
distances, indices = query_index(example_sentences, k=5)

#Step 5: Retrieve and Display Results
for i, sentence in enumerate(example_sentences):
    print(f"Query: {sentence}")
    for j in range(len(indices[i])):
        doc_id = doc_ids[indices[i][j]]
        print(f"DocID: {doc_id}, Distance: {distances[i][j]}")

#exit()

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

def output_results_to_file(queries, doc_ids, output_filepath, k=10, tiptoe = False):
    """
    Process each query, generate embeddings, make the query and output the results to a file.
    """
    max_query_length = 100
    print("Processing", max_query_length, "queries")

    start = time.time()
    query_embeddings = generate_query_embeddings([sentence for question_id, sentence in queries][:max_query_length])
    end = time.time()
    print("Time to generate query embeddings: ", end - start)
    print("Average time to generate each query embedding: ", (end - start)/max_query_length)

    print(embeddings.shape) 


    # measure the time to process the queries
    start = time.time()
    current_time = datetime.now().strftime("%H:%M:%S")
    print(current_time + ": Processing queries...")
    with open(output_filepath, 'w', encoding='utf-8') as outfile:
        t = 0
        for question_id, sentence in queries:
            #query_embedding = model.encode([sentence])  # Note: [sentence] to keep input as batch
            #distances, indices = query_index(sentence, k)
            distances, indices = find_approximate_nearest_neighbors(query_embeddings[t], k, tiptoe=tiptoe)
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
            if t == max_query_length:
                break
    end = time.time()
    current_time = datetime.now().strftime("%H:%M:%S")
    print(current_time + ":Processed", t,  " queries")

    # measure the time to process the queries in seconds
    print("Time to process the queries: ", end - start)
    print("Average time to process each query: ", (end - start)/max_query_length)


query_filepath = 'msmarco-queries-1000.tsv'
result_filepath = 'cluster-msmarco-results.txt'
queries = read_queries(query_filepath)
output_results_to_file(queries[0:100], doc_ids, result_filepath, k = 100, tiptoe=False)
