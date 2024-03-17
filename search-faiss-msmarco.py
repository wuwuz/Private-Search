import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime
import os

from cuml.decomposition import PCA as cuPCA
from cuml.decomposition import IncrementalPCA as cuIPCA
import cudf
import pickle
import time

DIM_REDUCED = False

# Step 1: Load DocIDs and Embeddings
def load_data(docid_filepath, embeddings_filepath):
    # Load DocIDs
    #with open(docid_filepath, 'r') as f:
    #    doc_ids = [line.strip() for line in f.readlines()]

    doc_ids = np.load(docid_filepath)
    #print(doc_ids)

    # Load embeddings
    embeddings = np.load(embeddings_filepath)
    print(embeddings.shape)
    
    return doc_ids, embeddings

# Initialize your model for embeddings
model = SentenceTransformer('msmarco-distilbert-base-tas-b')

# Import 

# Step 2: Create and Populate a FAISS Index
def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]  # Assuming 768-dimensional embeddings
    # use faiss hnsw to create the index
    #index = faiss.IndexHNSWFlat(dimension, 32)
    index = faiss.IndexHNSWFlat(dimension, 32)
    #index = faiss.IndexFlatIP(dimension) 
    index.add(embeddings)  # Adding the embeddings to the index
    return index

# Step 3: Generate Query Embeddings
def generate_query_embeddings(sentences):
    tmp = model.encode(sentences)
    if DIM_REDUCED:
        #print("Transforming query embeddings with PCA model...")
        tmp = ipca.transform(tmp)
    return tmp
    #return model.encode(sentences)

# Step 4: Query the FAISS Index
def query_index(index, query_embeddings, k=10):
    # k is the number of nearest neighbors you want to retrieve
    distances, indices = index.search(query_embeddings, k)
    return distances, indices

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

def output_results_to_file(queries, query_embeddings, index, model, doc_ids, output_filepath, k=10):
    """
    Process each query, generate embeddings, query the FAISS index, and output the results to a file.
    """

    #start = time.process_time() 
    #end = time.process_time()
    #print("Time to generate query embeddings: ", end - start)
    #print("Average time to generate each query embedding: ", (end - start)/len(queries))

    vector_dim = query_embeddings.shape[1]


    start = time.process_time()
    with open(output_filepath, 'w', encoding='utf-8') as outfile:
        t = 0
        for question_id, sentence in queries:
            t += 1
            if t % 10 == 0:
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                print(current_time + ":Processed", t,  " queries")
            #query_embedding = model.encode([sentence])  # Note: [sentence] to keep input as batch
            #query_embedding = generate_query_embeddings([sentence])
            distances, indices = index.search(query_embeddings[t - 1].reshape(1,vector_dim), k)
            
            outfile.write(f"Query: {question_id} {sentence}\n")
            for i in range(k):
                doc_id = doc_ids[indices[0][i]]
                distance = distances[0][i]
                outfile.write(f"{doc_id}\n")
            outfile.write("----------\n\n")
    end = time.process_time()
    print("Time to process the queries: ", end - start)
    print("Average time to process each query: ", (end - start)/len(queries))

# Example Usage
#docid_filepath = 'msmarco1000_docid.npy'
#embeddings_filepath = 'msmarco1000_embeddings.npy'  # Assuming .npy format for simplicity
docid_filepath = 'msmarco_docid_permuted.npy'
embeddings_filepath = 'msmarco_embeddings_reduced_permuted.npy'  # Assuming .npy format for simplicity
query_filepath = 'msmarco-queries-1000.tsv'
result_filepath = 'faiss-msmarco-reduced-results.txt'

index_file = 'msmarco_hnsw_index_permuted.faiss'

doc_ids, embeddings = load_data(docid_filepath, embeddings_filepath)

if os.path.exists(index_file):
    print("Loading index from file")
    index = faiss.read_index(index_file)
else:
    print("Creating index...")
    index = create_faiss_index(embeddings)

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

#Example sentences to query
example_sentences = ["Physics is cool", "I love physics", "Physics is awesome", "Physics is boring"]
query_embeddings = generate_query_embeddings(example_sentences)
distances, indices = query_index(index, query_embeddings, k=5)

#Step 5: Retrieve and Display Results
for i, sentence in enumerate(example_sentences):
    print(f"Query: {sentence}")
    for j in range(len(indices[i])):
        doc_id = doc_ids[indices[i][j]]
        print(f"DocID: {doc_id}, Distance: {distances[i][j]}")

query_embedding_file = 'msmarco-queries-1000-embeddings.npy'
queries = read_queries(query_filepath)
# test if the query embedding file exists

if os.path.exists(query_embedding_file):
    print("Loading query embeddings from file")
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

queries = read_queries(query_filepath)
#query_embeddings = generate_query_embeddings([sentence for question_id, sentence in queries])
#output_results_to_file(queries, index, model, doc_ids, result_filepath, k = 50)
max_query_num = 1000#len(queries)

start = time.time()
output_results_to_file(queries[0:max_query_num], query_embeddings[0:max_query_num], index, model, doc_ids, result_filepath, k = 100)
end = time.time()
print("Time to process the queries: ", end - start)
print("Average time to process each query: ", (end - start)/max_query_num)
