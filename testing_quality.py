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

# Load the embeddings data
#embeddings_file = "msmarco_embeddings_reduced.npy"
#embeddings_file = "msmarco_embeddings_reduced.npy"
#embeddings = np.load(embeddings_file)

#docid_file = "msmarco_docid.npy"
docid_file = "msmarco_docid_permuted.npy"
doc_ids = np.load(docid_file)

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

def read_result(filepath, query_num, k):
    # the file contains the response from the go frontend
    # it has query_num lines, each with k integers, this function returns the matrix

    response = np.zeros((query_num, k), dtype = int)
    with open(filepath, 'r', encoding='utf-8') as file:
        for i in range(query_num):
            line = file.readline()
            response[i] = [int(x) for x in line.split()]

    return response

def vertex2docid(queries, response, doc_ids, output_filepath, max_query_num = 100, k = 10):
    """
    Process each query, generate embeddings, make the query and output the results to a file.
    """

    # measure the time to process the queries
    with open(output_filepath, 'w', encoding='utf-8') as outfile:
        t = 0
        for question_id, sentence in queries:
            #query_embedding = model.encode([sentence])  # Note: [sentence] to keep input as batch
            #distances, indices = query_index(sentence, k)
            indices = response[t]
            outfile.write(f"Query: {question_id} {sentence}\n")
            for i in range(len(indices)):    
                doc_id = doc_ids[indices[i]]
                outfile.write(f"{doc_id}\n")
            outfile.write("----------\n\n")
            t += 1
    
    print(f"Results have been written to {output_filepath}.")

max_query_num = 1000
k = 100

query_filepath = 'msmarco-queries-1000.tsv'
#result_filepath = 'pir-msmarco-results.txt'
#frontend_result_filepath = 'go_frontend_result.txt'
#result_filepath = 'go_frontend_result_docid.txt'
#frontend_result_filepath = 'msmarco-dataset/msmarco_embeddings_3201821_192_32_output.txt'
#result_filepath = 'msmarco-dataset/msmarco_embeddings_3201821_192_32_output_docid.txt'
frontend_result_filepath = 'msmarco-dataset/cluster_search_result.txt'
result_filepath = 'msmarco-dataset/cluster_search_result_docid.txt'

queries = read_queries(query_filepath)
queries = queries[:max_query_num]
response = read_result(frontend_result_filepath, max_query_num, k)
vertex2docid(queries, response, doc_ids, result_filepath, max_query_num, k)





# clock the total time

# use system call to call a python script named "mrr.py"
os.system("python mrr.py " + result_filepath)
#os.system("python mrr.py " + result_filepath + ">> result-priv-ann.txt")
