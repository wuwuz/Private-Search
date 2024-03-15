import numpy as np
import faiss
import datetime
import os
import heapq

NEIGHBORHOOD_SIZE = 32
TARGET_NEIBHORHOOD_SIZE = 32
TESTING = False
TESTING_SIZE = 100000
# set random seed
np.random.seed(0)

# Load the embedding vectors
embeddings = np.load('msmarco_embeddings_reduced_permuted.npy')

# for testing
if TESTING:
    embeddings = embeddings[:TESTING_SIZE]

INDEX_FILE = 'msmarco_hnsw_index_permuted.faiss'

# Build an HNSW index
dimension = 192  # Dimension of the vectors

# if the index file exists, load it
if os.path.exists(INDEX_FILE) and not TESTING:
    index = faiss.read_index(INDEX_FILE)
    print(f'Loaded index from {INDEX_FILE}')
else:
    index = faiss.IndexHNSWFlat(dimension, 32)  # 32 is the default value for M parameter in HNSW
    faiss.omp_set_num_threads(8)  # Adjust based on your system for potentially better performance
    index.add(embeddings)
    # save the index
    if not TESTING:
        faiss.write_index(index, INDEX_FILE)
        print(f'Saved index to {INDEX_FILE}')

#weird_idx = 2647078
#D, I = index.search(embeddings[weird_idx:weird_idx+1], 100)
#print("D: ", D)
#print("I: ", I)


# Define batch size
batch_size = 128

# Prepare a list to hold the indices of connected neighbors
connected_neighbors_list = []


# the function to select the neighbors
# center_id: the index of the vector
# candidates: the indices of the candidate neighbors
# target_num: the number of neighbors to select
def select_neighbors(center_id, candidates, target_num):
    # first sort the candidates based on the distance to the center vector
    candidates = sorted(candidates, key=lambda x: np.linalg.norm(embeddings[center_id] - embeddings[x]))
    center_vector = embeddings[center_id]
    connected_neighbors = []
    discarded_candidates = []
    for candidate in candidates:  # Skip the first index as it's the vector itself
        check = True
        candidate_vector = embeddings[candidate]
        candidate_distance = np.linalg.norm(center_vector - candidate_vector)
        for j in range(len(connected_neighbors)):
            connected_neighbor = connected_neighbors[j]
            connected_neighbor_vector = embeddings[connected_neighbor]

            # if the current neighbor is closer to the connected neighbor than the center_vector itself, then check = False
            distance_to_connected_neighbor = np.linalg.norm(candidate_vector - connected_neighbor_vector)
            if distance_to_connected_neighbor < candidate_distance:
                check = False
                break
        if check :
            connected_neighbors.append(candidate)
            if len(connected_neighbors) == target_num:
                break
        else:
            discarded_candidates.append(candidate)

    if len(connected_neighbors) < target_num:
        # append the discarded neighbors to the connected neighbors
        connected_neighbors.extend(discarded_candidates[:target_num - len(connected_neighbors)])
    
    return connected_neighbors

# Batch processing
num_vectors = embeddings.shape[0]
graph = [set([]) for _ in range(num_vectors)]

hnsw_candidate_mentioned_num = np.zeros(num_vectors, dtype=int)

# a set of existing edges



for i in range(0, num_vectors, batch_size):
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # print the batch info and time 
    #if TESTING == False:
    if i // batch_size % 100 == 0:
        print(f'{current_time} Processing batch {i // batch_size + 1} of {num_vectors // batch_size + 1}...')
    end = min(i + batch_size, num_vectors)
    D, I = index.search(embeddings[i:end], 150)  # Search for the 100 nearest neighbors (+1 for the vector itself)
    
    # Process each vector in the batch
    for j in range(D.shape[0]):#
        #if -1 in I[j]:
        #    print(f'Warning: vector {i+j} has -1 in its neighbors')
        #    print(f'Neighbors: {I[j]}')
        # current vector is embeddings[i+j]
        candidates = I[j]
        for candidate in candidates:
            hnsw_candidate_mentioned_num[candidate] += 1
        possible_candidates = []
        high_deg_candidates = []
        for possible_candidate in candidates:
            if possible_candidate == i+j or possible_candidate == -1:
                continue
            #if len(graph[possible_candidate]) < 1.5 * NEIGHBORHOOD_SIZE:
            possible_candidates.append(possible_candidate)  
            #else:
            #    high_deg_candidates.append(possible_candidate)
        #if len(possible_candidates) < NEIGHBORHOOD_SIZE:
            #possible_candidates.extend(high_deg_candidates[:min(len(high_deg_candidates), NEIGHBORHOOD_SIZE - len(possible_candidates))])
        #if len(possible_candidates) < NEIGHBORHOOD_SIZE:
        #    possible_candidates.extend(high_deg_candidates[:NEIGHBORHOOD_SIZE - len(possible_candidates)])
        connected_neighbors = select_neighbors(i+j, possible_candidates, NEIGHBORHOOD_SIZE)
        for connected_neighbor in connected_neighbors:
            if connected_neighbor not in graph[i + j]:
                graph[i + j].add(connected_neighbor)
                graph[connected_neighbor].add(i + j)

    #if i // batch_size == 10: 
    #    break

# find the top 100 mentioned idx in hnsw_candidate_mentioned_num
#top_100_mentioned_idx = np.argsort(hnsw_candidate_mentioned_num)[::-1][:100]
#print("Top 100 mentioned idx: ", top_100_mentioned_idx)
# also print their mentioned num
#print("Top 100 mentioned num: ", hnsw_candidate_mentioned_num[top_100_mentioned_idx])

# find the least 100 mentioned idx in hnsw_candidate_mentioned_num
#top_100_mentioned_idx = np.argsort(hnsw_candidate_mentioned_num)[:100]
#print("Least 100 mentioned idx: ", top_100_mentioned_idx)
# also print their mentioned num
#print("Least 100 mentioned num: ", hnsw_candidate_mentioned_num[top_100_mentioned_idx])

#mentioned_num_hist = np.bincount(hnsw_candidate_mentioned_num)[0:200]
#print("sum of those mentioned num: ", np.sum(mentioned_num_hist))
#for i in range(len(mentioned_num_hist)):
#    print(f'Number of vectors that are mentioned {i} times: {mentioned_num_hist[i]}')

out_degree = [len(graph[i]) for i in range(num_vectors)]
print("Min out degree: ", np.min(out_degree))
print("Max out degree: ", np.max(out_degree))

#degree = [len(graph[i]) for i in range(num_vectors)]
degree = np.zeros(num_vectors, dtype=int)
for i in range(num_vectors):
    for j in graph[i]:
        degree[j] += 1

print("Min in degree: ", np.min(degree))
print("Max in degree: ", np.max(degree))

#print("graph 2647078", graph[2647078])

bad_edges_num = 0
# now trim the graph to have exactly TARGET_NEIGHBORHOOD_SIZE neighbors for each vector
for i in range(num_vectors):
    if i % 10000 == 0:
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # print time and the vector index
        print(f'{current_time} Processing vector {i}...')
    # remove the duplicates
    raw_candidates = list(set(graph[i]))
    candidates = []

    for candidate in raw_candidates:
        # sample the candidates with prob. 2 * TARGET_NEIBHORHOOD_SIZE / degree[candidate]
        sample_prob = 2.0 * TARGET_NEIBHORHOOD_SIZE / degree[candidate]
        if np.random.rand() < sample_prob:
            candidates.append(candidate)

    #graph[i] = select_neighbors(i, candidates, TARGET_NEIBHORHOOD_SIZE) 
    #if len(graph[i]) < TARGET_NEIBHORHOOD_SIZE:
        #print(f'Warning: vector {i} has only {len(graph[i])} neighbors')
        # append random neighbors to the graph[i]
    #    graph[i].extend(np.random.choice(num_vectors, TARGET_NEIBHORHOOD_SIZE - len(graph[i])))

    # just randomly select the target number of neighbors
    if len(candidates) >= TARGET_NEIBHORHOOD_SIZE:
        #graph[i] = np.random.choice(candidates, TARGET_NEIBHORHOOD_SIZE)
        graph[i] = select_neighbors(i, candidates, TARGET_NEIBHORHOOD_SIZE)
    else: 
        graph[i] = candidates
        bad_edges_num += TARGET_NEIBHORHOOD_SIZE - len(candidates)
        added_random = np.random.choice(num_vectors, TARGET_NEIBHORHOOD_SIZE - len(candidates))
        #print(added_random)
        graph[i].extend(added_random)

    # finally, sort the neighbors based on their distance to the current vector
    current_vector = embeddings[i]
    graph[i] = sorted(graph[i], key=lambda x: np.linalg.norm(current_vector - embeddings[x]))

    # find the neighbor with the max index
    max_idx = np.max(graph[i])
    if max_idx < num_vectors / 3:
        print("Warning: max_idx < num_vectors / 3")
        print("max_idx: ", max_idx)
        print("graph[i]: ", graph[i])

print("Bad edges num: ", bad_edges_num)
print("ratio of bad edges: ", bad_edges_num / (num_vectors * TARGET_NEIBHORHOOD_SIZE))

graph = np.array(graph, dtype=int)

#for i in range(100, 200):
#    print("graph ", i, " ==", graph[i])

# select 0...sqrt(n) as the representative vectors
#reps = np.random.choice(num_vectors, int(np.sqrt(num_vectors)), replace=False)
reps = np.arange(0, num_vectors, int(np.sqrt(num_vectors)))
#print("reps: ", reps)
rep_vector = embeddings[reps]

def find_approximate_nearest_neighbors(query_vector, k = 100, step = 10, hidden_id = None):
    hit_step = 99999

    # first find the representative vector that is closest to the query vector, based on dot product similarity
    distances = np.linalg.norm(rep_vector - query_vector, axis=1) #this is based on L2
    #print("distance to the rep vectors: ", distances)
    entry_distance = np.min(distances)
    entry_idx = reps[np.argmin(distances)]

    #entry_idx = 1
    entry_distance = np.linalg.norm(query_vector - embeddings[entry_idx])
    visited = set([entry_idx])

    if entry_idx == hidden_id:
        hit_step = 0

    # the to_be_explored list is a priority queue containing tuples
    # each tuples contains the distance to the query_vector and the index

    to_be_explored = []
    heapq.heappush(to_be_explored, (entry_distance, entry_idx))

    for i in range(1, step + 1):
        #print("Step: ", i)
        #print("to_be_explored: ", to_be_explored)
        #print("graph[to_be_explored]: ", graph[to_be_explored])
        current_distance, current_idx = heapq.heappop(to_be_explored)
        #print("current_idx: ", current_idx)
        #print("current_distance: ", current_distance)
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


# compute the in-degree of graph

in_degree = np.zeros(num_vectors, dtype=int)
for i in range(num_vectors):
    for j in graph[i]:
        in_degree[j] += 1

# The in_degree array stores each vertex's in-degree. Now print the 100 indices with least in_degree
sorted_in_degree_with_idx = np.argsort(in_degree)
print("The 100 indices with least in_degree: ", sorted_in_degree_with_idx[:100])
print("The 100 in_degree: ", in_degree[sorted_in_degree_with_idx[:100]])


min_in_degree = np.min(in_degree[1:])
min_id = np.argmin(in_degree[1:])
max_in_degree = np.max(in_degree[1:])
max_id = np.argmax(in_degree[1:])
print("Min in degree: ", min_in_degree, " at ", min_id - 1)
print("Max in degree: ", max_in_degree, " at ", max_id - 1)

# test recall
max_step = 20
hit_step_history = []
for i in range(len(reps) + 1, len(reps) + 101):
    #print("Query: ", i)
    distances, indices, hit_step = find_approximate_nearest_neighbors(embeddings[i], k=10, step=max_step, hidden_id=i)
    if hit_step == 99999:
        hit_step = max_step + 1
    #print("Query: ", i, " hit_step: ", hit_step)
    hit_step_history.append(hit_step)

histogram = np.bincount(hit_step_history)
print("Average hit step: ", sum(hit_step_history)/len(hit_step_history))
histogram = histogram / np.sum(histogram)
print("Histogram of hit steps: ", histogram)
print("ratio of hit", np.sum(histogram[:max_step]))


# convert the connected_neighbors_list to a numpy array
# each row will have exactly NEIGHBORHOOD_SIZE elements
connected_neighbors_indices = np.array(graph, dtype=int)
print(connected_neighbors_indices.shape)
if not TESTING:
    np.save('graph_permuted.npy', connected_neighbors_indices)

    # also read the reps id and print their neighbors to a file
    reps = np.load('msmarco-1280-cluster-reps-relabeled.npy')
    # now the reps are the indices of the representatives. Save their neighbors to an npy file
    reps_neighbors = graph[reps]
    np.save('msmarco-1280-reps-neighbors-relabeled.npy', reps_neighbors)

    # also save the graph as a text file
    with open('graph_permuted.txt', 'w') as file:
        for i in range(connected_neighbors_indices.shape[0]):
            file.write(' '.join(map(str, connected_neighbors_indices[i])) + '\n')