import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime

from cuml.decomposition import PCA as cuPCA
from cuml.decomposition import IncrementalPCA as cuIPCA
import cudf
import pickle
import time


OUTPUT_FILE = "cluster-msmarco-PIANO-results.txt"
# CLUSTER_NUM = 4 * 32 * 10
CLUSTER_NUM = 100 # for testing, and wsl doesnt have enough memory
# DIM_REDUCED = False


# Initialize your model for embeddings
model = SentenceTransformer('msmarco-distilbert-base-tas-b')

# Load the embeddings data
embeddings_file = "msmarco_embeddings_reduced.npy"
embeddings = np.load(embeddings_file)

docid_file = "msmarco_docid.npy"
doc_ids = np.load(docid_file)

# Load the cluster representatives
representatives_file = "msmarco-" + str(CLUSTER_NUM) + "-cluster-reps.npy"
reps = np.load(representatives_file)

# Load the cluster labels for each embedding vector
labels_file = "msmarco-" + str(CLUSTER_NUM) + "-cluster-labels.npy"
labels = np.load(labels_file)
# print("labels: ", labels)

# sort the vectors, so that the vectors in the same cluster are grouped together
sorted_indices = np.argsort(labels)
#sorted_labels = labels[sorted_indices]
sorted_embeddings = embeddings[sorted_indices]

# now decide the start and end of each cluster
num_vector_in_each_cluster = np.bincount(labels)

# now build a prefix sum use pure numpy operation
start_idx = np.cumsum(num_vector_in_each_cluster)
start_idx = np.concatenate(([0], start_idx))

# print("This is the number of items in each cluster: ", start_idx)
# print("Example of a specific embedding: ", sorted_embeddings[start_idx[0]:start_idx[1]])
max_cluster_size = np.max(num_vector_in_each_cluster)

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

def sample_clusters():
    #treat each cluster as a single element, suppose there are n clusters
    #we want to sample one random sqrt(n) indices from our n clusters as follows:
    #1. cluster our clusters further into sqrt(n) clusters
    #2. sample one random index from each of the sqrt(n) clusters
    #3. return the indices of the sampled clusters

    #cluster our clusters further into sqrt(n) clusters
    sqrt_num_clusters = int(np.sqrt(CLUSTER_NUM))
    #print("sqrt_num_clusters: ", sqrt_num_clusters)

    #sample one random index from each of the sqrt(n) clusters
    return set([i * sqrt_num_clusters + np.random.randint(sqrt_num_clusters) for i in range(sqrt_num_clusters)])

#after sampling the clusters for a set of size sqrt(n), compute the parity of the set.
#each element in sampled_clusters is going to be a vector of size sqrt(n), where it should be bitwise xor'd
def compute_parity(sampled_clusters):
    parity = np.zeros((max_cluster_size, embeddings.shape[1]), dtype=np.float64)
    parity = parity.view(np.int64)
    for i in sampled_clusters:
        start, end = start_idx[i], start_idx[i+1]
        cluster = sorted_embeddings[start:end]
        #add dummy zero vectors of the embedding size to cluster if less than max size
        if cluster.shape[0] < max_cluster_size:
            cluster = np.vstack([cluster, np.zeros((max_cluster_size - cluster.shape[0], cluster.shape[1]))])
        
        # View the floats as 64-bit integers
        cluster = cluster.view(np.int64)
        
        # #compute online parity
        parity = np.bitwise_xor(parity, cluster)
    #convert back to float
    parity = parity.view(np.float64)
    return parity


# sampled_clusters = sample_clusters()
# print("sampled_clusters: ", sampled_clusters)
# parity = compute_parity(sampled_clusters)
# print("parity: ", parity)

#run sample_cluster sqrt(n) times and compute the parity of each set
#store the parity of each set in a list
def compute_primary():
    res = []
    for i in range(int(np.sqrt(CLUSTER_NUM))):
        sampled_clusters = sample_clusters()
        parity = compute_parity(sampled_clusters)
        res.append((sampled_clusters,parity))
    return res

def compute_replacement_entries():
    #for each sqrt(n) of n clusters, choose a random index from that cluster and its embedding vector
    #store the index and the embedding vector in a list
    replacement_entries = []
    for i in range(int(np.sqrt(CLUSTER_NUM))):
        cluster_id = i * int(np.sqrt(CLUSTER_NUM)) + np.random.randint(np.sqrt(CLUSTER_NUM))
        start,end = start_idx[cluster_id], start_idx[cluster_id + 1]
        
        #pad the cluster with zero vectors if less than max size
        if end - start < max_cluster_size:
            cluster = np.vstack([sorted_embeddings[start:end], np.zeros((max_cluster_size - (end - start), sorted_embeddings.shape[1]))])
        else:
            cluster = sorted_embeddings[start:end]
        replacement_entries.append((cluster_id, cluster))
    return replacement_entries

def find_approximate_nearest_neighbors(query_vector, k = 100,tiptoe = False):
#     # first find the representative vector that is closest to the query vector, based on dot product similarity
    distances = np.linalg.norm(rep_vector - query_vector, axis=1) #this is based on L2
#     #print("distance to the rep vectors: ", distances)
    cluster_idx = np.argmin(distances)
    start, end = start_idx[cluster_idx], start_idx[cluster_idx + 1]
    
    if not tiptoe:    
        primary_table = compute_primary()
        replacement_entries = compute_replacement_entries() 
        for S,p in primary_table:
            if cluster_idx in S:
                j = int(cluster_idx // np.sqrt(CLUSTER_NUM))
                r,db_r = replacement_entries[j]
                #replace cluster_idx in S with r
                S.remove(cluster_idx)
                S.add(r)
                #calculate the parity of the new S
                p_prime = compute_parity(S)
                p_prime,db_r,p = p_prime.view(np.int64), db_r.view(np.int64), p.view(np.int64)
                result = np.bitwise_xor(db_r,np.bitwise_xor(p, p_prime)).view(np.float64)
                # print("expected: ", sorted_embeddings[start:end][:3])
                
                #result should now be the correct cluster, return the top k nearest vectors in the cluster
                #dot product absolute value 
                distance_to_cluster = np.dot(result, query_vector)
                #make sure all are positive
                distance_to_cluster = np.abs(distance_to_cluster)
            
                if end - start < k:
                    top_k_idx = np.argsort(distance_to_cluster[:end - start])[::-1]
                else:
                    top_k_idx = np.argsort(distance_to_cluster)[::-1][:k]
                original_idx = sorted_indices[start:end][top_k_idx]
                distances = distance_to_cluster[top_k_idx]
                return distances, original_idx
        
    #otherwise default to tiptoe impl.
    distance_to_cluster = np.dot(sorted_embeddings, query_vector)[start:end]

    top_k_idx = np.argsort(distance_to_cluster)[::-1][:k]
    
    original_idx = sorted_indices[start:end][top_k_idx]
    distances = distance_to_cluster[top_k_idx]
    return distances, original_idx



def query_index(sentence, k = 10,tiptoe=False):
    if isinstance(sentence, str):
        sentence = [sentence]
    embeddings = generate_query_embeddings(sentence)
    result_distance = []
    result_idx = []
    for i in range(len(embeddings)):
        distances, original_idx = find_approximate_nearest_neighbors(embeddings[i], k,tiptoe)
        result_distance.append(distances)
        result_idx.append(original_idx)
    return result_distance, result_idx

#Example sentences to query - Returns the same results as tiptoe processing (correctness check)
# example_sentences = ["derriere definition"]
#query_embeddings = generate_query_embeddings(example_sentences)
# distances1, indices1 = query_index(example_sentences, k=5,tiptoe=True)
# distances2, indices2 = query_index(example_sentences, k=5,tiptoe=False)
#Step 5: Retrieve and Display Results
# for i, sentence in enumerate(example_sentences):
#     print(f"Query: {sentence}")
#     for j in range(len(indices2[i])):
#         doc_id1 = doc_ids[indices1[i][j]]
#         doc_id2 = doc_ids[indices2[i][j]]
#         print(f"TIPTOE DocID: {doc_id1},  Distance: {distances1[i][j]}")
#         print(f"PIANO DocID: {doc_id2},  Distance: {distances2[i][j]}")
        

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


    # measure the time to process the queries
    start = time.time()
    current_time = datetime.now().strftime("%H:%M:%S")
    print(current_time + ": Processing queries...")
    with open(output_filepath, 'w', encoding='utf-8') as outfile:
        t = 0
        for question_id, sentence in queries:
            #query_embedding = model.encode([sentence])  # Note: [sentence] to keep input as batch
            # print(f"Query: {question_id} {sentence}")
            # distances, indices = query_index(sentence,k=k,tiptoe=False)
            
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
queries = read_queries(query_filepath)
output_results_to_file(queries[0:100], doc_ids, OUTPUT_FILE, k = 10, tiptoe=False)

# queries = read_queries(query_filepath)
output_results_to_file(queries[0:100], doc_ids, "cluster-msmarco-TIPTOE-results.txt", k = 10, tiptoe=True)




