import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime
import heapq
import os
import requests

#from cuml.decomposition import PCA as cuPCA
#from cuml.decomposition import IncrementalPCA as cuIPCA
#import cudf
import pickle
import time
import subprocess

CLUSTER_NUM = 4 * 32 * 10
#CLUSTER_NUM = 100
DIM_REDUCED = False

# Initialize your model for embeddings
model = SentenceTransformer('msmarco-distilbert-base-tas-b')

# Load the embeddings data
#embeddings_file = "msmarco_embeddings_reduced.npy"
#embeddings_file = "msmarco_embeddings_reduced.npy"
#embeddings = np.load(embeddings_file)

#docid_file = "msmarco_docid.npy"
docid_file = "msmarco_docid_permuted.npy"
doc_ids = np.load(docid_file)

# Load the cluster representatives
representatives_file = "msmarco-" + str(CLUSTER_NUM) + "-cluster-reps-relabeled.npy"
reps = np.load(representatives_file)

# now extract the representative vectors for each cluster, based on the representative indices
#rep_vector = embeddings[reps]
rep_vector_file = "msmarco-" + str(CLUSTER_NUM) + "-reps-vector.npy"
rep_vector = np.load(rep_vector_file)

# Load the graph data
#graph_file = "connected_neighbors.npy"
#graph_file = "015graph.npy"
graph_file = "graph_permuted.npy"
graph = np.load(graph_file)

print("graph shape: ", graph.shape)

#print(graph[267078])

#exit()

# Load the rep graph data
rep_graph_file = "msmarco-" + str(CLUSTER_NUM) + "-reps-neighbors-relabeled.npy"
rep_graph = np.load(rep_graph_file)

# verify the rep_graph matches the graph
#for i in range(len(reps)):
#    rep = reps[i]
#    check = True
#    for j in range(len(rep_graph[i])):
#        if rep_graph[i][j] != graph[rep][j]:
#            check = False
#            break
#    if check == False:
#        print("Error: rep_graph does not match graph")
#        print("rep: ", rep, "rep_graph: ", rep_graph[i], "graph: ", graph[rep])
#        break
#exit()


# load the PCA model
#with open('ipca_msmarco.pkl', 'rb') as f:
#    ipca = pickle.load(f)

# check if the number of components in the PCA model is the same as the number of dimensions in the embeddings
'''
if rep_vector.shape[1] == ipca.n_components:
    # transform the query embeddings
    print("Transforming query embeddings with PCA model...")
    #query_embeddings = ipca.transform(query_embeddings)
    DIM_REDUCED = True
else:
    DIM_REDUCED = False
    print("No need to transform")
'''

def generate_query_embeddings(sentences):
    tmp = model.encode(sentences)
    #if DIM_REDUCED:
        #print("Transforming query embeddings with PCA model...")
    #    tmp = ipca.transform(tmp)
    return tmp
    #return model.encode(sentences)

def query_PIR_info():
    response = requests.get('http://localhost:8080/info')
    if response.status_code == 200:
        data = response.json()

        DBSize = data['DBSize']
        PrepTime = data['PrepTime']
        Storage = data['Storage']
        OnlineComm = data['OnlineComm']
        OfflineComm = data['OfflineComm']

        print("-----------------------")
        print("PIR Info: ")
        print("DBSize (MB): ", DBSize / 1024 / 1024)
        print("PrepTime (seconds): ", PrepTime)  
        print("Storage: (MB)", Storage / 1024 / 1024)
        print("OnlineCommPerBatch (KB): ", OnlineComm / 1024)
        print("OfflineCommPerBatch (KB): ", OfflineComm / 1024)
        print("-----------------------")

        # return a tuple containing the PIR info
        return (DBSize, PrepTime, Storage, OnlineComm, OfflineComm)
    else:
        print("Error:", response.status_code)


class VertexInfo:
    # a list containing the indices of the neighbors of the vertex
    def __init__(self, index, vector, neighbors, flag = True, from_cache = False):
        self.index = index
        self.vector = vector
        self.neighbors = neighbors
        self.flag = flag

        # verify the correctness
        # ignore now
        #if self.flag == True:
        #    check = True
        #    for neighbor in self.neighbors:
        #        if neighbor not in graph[self.index]:
        #            check = False
        #            break

        #    if check == False:
        #        print("Read Graph Info Error: ", self.index, self.neighbors, graph[self.index])
        #        print(len(self.neighbors), len(graph[self.index]))
        #        print("from cache: ", from_cache)
            #else:
                #print("Read Graph Info Correct:", self.index)

def query_vertex_info(indices, explored_graph, explored_vector):

    VertexInfoList = []
    to_query_list = []

    for idx in indices:
        if idx in explored_vector:
            VertexInfoList.append(VertexInfo(idx, explored_vector[idx], explored_graph[idx], from_cache=True))
        else: 
            to_query_list.append(idx)
            query_string = '&'.join([f'rowIndex={idx}' for idx in indices])
    
    if len(to_query_list) == 0:
        return VertexInfoList

    query_string = '&'.join([f'rowIndex={idx}' for idx in to_query_list])
    #print("query_string: ", query_string)
    response = requests.get(f'http://localhost:8080/query?{query_string}')

    if response.status_code == 200:
        data = response.json()

        for item, index in zip(data, to_query_list):
            #print(f"Row: {item['matrixRow']}, Neighbors: {item['neighbors']}")
            flag = (item['neighbors'][0] != 0 or item['neighbors'][1] != 0) # if the neighbors list is all zeros, then the query failed and the flag is False 
            VertexInfoList.append(VertexInfo(index, item['matrixRow'], item['neighbors'], flag))
        return VertexInfoList
    else:
        print("Error:", response.status_code)
        return None



# query_vector: the query vector
# k: The number of nearest neighbors to return
# step: The total rounds of exploration 
# parallel_exploration: The number of parallel vertices to explore in each round
# hidden_id: Only used for testing when search an existing vector in the database. 
#            Used to test the efficiency of the search algorithm. 
#            The earlier we can find the hidden_id, the more efficient the search algorithm is.

def find_approximate_nearest_neighbors(query_vector, k = 100, step = 10, parallel_exploration = 1, hidden_id = None):
    hit_step = 99999

    # first find the representative vector that is closest to the query vector, based on dot product similarity
    distances = np.linalg.norm(rep_vector - query_vector, axis=1) #this is based on L2
    #print("distance to the rep vectors: ", distan

    # find the two closest representatives
    # first sort the indices based on the distance
    # then select the first two indices

    sorted_indices = np.argsort(distances)
    visited = set()
    explored_graph = {}
    explored_vector = {}

    for i in range(1):
    #entry_distance = np.min(distances)
    #entry_distance 
    #entry_offset = np.argmin(distances)
        entry_offset = sorted_indices[i]
        entry_idx = reps[entry_offset]
        entry_distance = distances[entry_offset]

    # compare rep_graph[entry_offset] and graph[entry_idx]
    #for i in range(len(rep_graph[entry_offset])):
    #    if rep_graph[entry_offset][i] != graph[entry_idx][i]:
    #        print("Error: rep_graph does not match graph")
    #        print("rep_graph: ", rep_graph[entry_offset], "graph: ", graph[entry_idx])
    #        break

        visited.add(entry_idx)
        # add the vector of entry_idx to the explored_vector map
        explored_vector[entry_idx] = rep_vector[entry_offset]
        explored_graph[entry_idx] = rep_graph[entry_offset]
    #explored_vector = {entry_idx: rep_vector[entry_offset]}
    # add the neighbors of the entry_idx to the to_be_explored list
    #explored_graph = {entry_idx: rep_graph[entry_offset]}

    # the to_be_explored list is a priority queue containing tuples
    # each tuples contains the distance to the query_vector and the index

    to_be_explored = []
    heapq.heappush(to_be_explored, (entry_distance, entry_idx))

    total_query = 0
    succ_query = 0
    cached_query = 0

    for i in range(step):
        #print("Step: ", i)
        #print("to_be_explored: ", to_be_explored)
        #print("graph[to_be_explored]: ", graph[to_be_explored])
        dist, current_idx = heapq.heappop(to_be_explored)
        print("current_idx: ", current_idx)
        print("current dist square", dist * dist)
        #print("in step ", i, "current_idx: ", current_idx, "current_neighbors size", len(explored_graph[current_idx]))
        # let neighbors be 
        current_neighbors = explored_graph[current_idx]
        #print("current neighbors: ", current_neighbors)

        #print("in step ", i, "current_idx: ", current_idx, "current_neighbors size", len(current_neighbors))
        vertex_info = query_vertex_info(current_neighbors, explored_graph, explored_vector) # TODO: optimize it so that repeated queries are not made


        for j in range(parallel_exploration - 1):
            
            if len(to_be_explored) == 0:  # there's nothing to be explored.
                break

            _, current_idx_2 = heapq.heappop(to_be_explored)
            current_neighbors_2 = explored_graph[current_idx_2]

            vertex_info_2 = query_vertex_info(current_neighbors_2, explored_graph, explored_vector)
            vertex_info = vertex_info + vertex_info_2

        #print("in step ", i, "retrieving ", len(vertex_info))

        #sorted_neighbors = sorted(current_neighbors)


        #succ_query_list = []

        #succ_query_list.append(vertex.index) 
        #print("in step ", i, "current_idx: ", current_idx, "current_neighbors size", len(current_neighbors), "sorted_neighbors: ", sorted_neighbors)
        #print("succ query list: ", succ_query_list)

        for vertex in vertex_info:
            total_query += 1
            if vertex.flag == False:
                continue

            if vertex.index in visited:
                cached_query += 1
                continue
            else:
                succ_query += 1

            visited.add(vertex.index)
            explored_graph[vertex.index] = vertex.neighbors
            explored_vector[vertex.index] = vertex.vector
            distance = np.linalg.norm(query_vector - vertex.vector)
            heapq.heappush(to_be_explored, (distance, vertex.index))
            if hidden_id != None and vertex.index == hidden_id:
                hit_step = i
    
    # now find the top k nearest neighbors
    # we insert all the explored idx and their distances into a list
    visited_tuples = []
    for (i, vector) in explored_vector.items():
        distance = np.linalg.norm(query_vector - vector)
        visited_tuples.append((distance, i))
    
    # sort the list based on the distance
    visited_tuples.sort(key = lambda x: x[0])

    # now return the top k nearest neighbors, their distances and the step when the hidden_id is found
    distances = [x[0] for x in visited_tuples[:k]]
    ids = [x[1] for x in visited_tuples[:k]]

    #if total_query > 0:
    #    print("Total Query: ", total_query, "Succ Query: ", succ_query,"ratio: ", succ_query/total_query, "Cached Query: ", cached_query, "ratio: ", cached_query/total_query)

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

def output_results_to_file(queries, query_embeddings, doc_ids, output_filepath, k=10, step = 20, parallel_exploration = 1):
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
            distances, indices, _ = find_approximate_nearest_neighbors(query_embeddings[t], k, step = step, 
                                                                       parallel_exploration=parallel_exploration,
                                                                       hidden_id = None)
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

#try_query = query_vertex_info([4, 5, 6, 7], {}, {})
#exit()

def generate_report(report_file_name, result_file_name, query_num, k, step, parallel_exploration, rtt, total_time):

    (DBSize, PrepTime, Storage, OnlineComm, OfflineComm) = query_PIR_info()

    # oepn the report file in appending mode
    with open(report_file_name, 'a') as file:
        file.write("-------------------------\n")
        file.write("MSMARCO Report\n")
        file.write("Settings:\n")
        file.write("** Vector Num:" + str(graph.shape[0]) + "\n")
        file.write("** DB Size (MB): " + str(DBSize / 1024 / 1024) + "\n")
        file.write("** Top K: " + str(k) + "\n")
        file.write("** Rounds: " + str(step) + "\n")
        file.write("** Parallel Exploration: " + str(parallel_exploration) + "\n")
        file.write("** RTT (s): " + str(rtt) + "\n")
        file.write("\n")
        file.write("Preprocessing Cost:\n")
        file.write("** Storage (MB): " + str(Storage / 1024 / 1024) + "\n")
        file.write("** Preparation Time (s): " + str(PrepTime) + "\n")
        file.write("** Offline Communication Cost Per Q (KB, amt.): " + str(OfflineComm * step * parallel_exploration / 1024) + "\n")
        file.write("\n")
        file.write("Online Cost:\n")
        avg_time = total_time / query_num
        file.write("** Average Computation Time Per Query (s): " + str(avg_time) + "\n")
        avg_time_with_rtt = avg_time + rtt * step
        file.write("** Average Total Time Per Q (s): " + str(avg_time_with_rtt) + "\n")
        file.write("** Online Communication Per Q (KB): " + str(OnlineComm * step * parallel_exploration/ 1024) + "\n")
        file.write("-----------------------\n")

    # use system call to call a python script named "mrr.py"
    os.system("python mrr.py " + result_file_name + ">> " + report_file_name)
    


query_filepath = 'msmarco-queries-1000.tsv'
#result_filepath = 'pir-msmarco-results.txt'
result_filepath = 'pir-msmarco-results-tmp.txt'
query_embedding_file = 'msmarco-queries-1000-embeddings.npy'
report_file = 'result-priv-ann.txt'

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

max_query_num = 1
k = 100
step = 15
parallel_exploration = 1
rtt = 0.05 # 50ms

# clock the total time
start = time.time()
output_results_to_file(queries[:max_query_num], query_embeddings[:max_query_num], doc_ids, result_filepath, 
                       k, step, parallel_exploration)
end = time.time()
print("Total Time: ", end - start)
print("Average Time: ", (end - start)/max_query_num)
#output_results_to_file(queries[2:3], query_embeddings[2:3], doc_ids, result_filepath, k = 100, step = 20)


generate_report(report_file, result_filepath,
                max_query_num, k, step, parallel_exploration, rtt, end-start) 

# use system call to call a python script named "mrr.py"
#os.system("python mrr.py " + result_filepath)
#os.system("python mrr.py " + result_filepath + ">> result-priv-ann.txt")
