import os
import sys
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from datetime import datetime

import pickle
import numpy
import time

#from config import *

SEQ_LEN = 512
NTHREADS = 1
#NTHREADS = max(1, multiprocessing.cpu_count() - 1)
#DATA_FILE = "test-prefix.tsv"
#DATA_FILE = "msmarco-prefix-1000.tsv"
DATA_FILE = "msmarco-prefix.tsv"

def save(entries, outfile):
    with open(outfile, 'wb') as f:
        pickle.dump(entries, f)

def compute_embeddings(model):
    model = SentenceTransformer(model)
    model.max_seq_length = SEQ_LEN
    lines = open(DATA_FILE).read().splitlines()
    print("Read all lines")
    sys.stdout.flush()
    new_data = [line.split('\t') for line in lines]
    print(new_data[0])

    print("Split all lines by tabs")
    sys.stdout.flush()
    #print([elem[2] for elem in new_data])
    chunked_data = [' '.join(elem[1].split()[0:SEQ_LEN]) for elem in new_data]
    print(chunked_data[0])
    print(len(chunked_data))
    print(len(chunked_data[0:32]))
    
    step = 1024
    #embeddings = numpy.array(model.encode(chunked_data[0:step], batch_size=32, convert_to_numpy=True))
    # create a zero array to concatenate the embeddings
    embeddings = numpy.zeros((0, 768), dtype = numpy.float32)
    print(embeddings.shape)
    for i in range(0, len(chunked_data), step):
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        up = min(i + step, len(chunked_data))
        print(current_time, "Encoding %d to %d" % (i, up))
        tmp = numpy.array(model.encode(chunked_data[i:up], batch_size=32, convert_to_numpy=True))
        print(tmp.shape)
        # concatenate the embeddings
        embeddings = numpy.concatenate((embeddings, tmp), axis=0)
    print(embeddings.shape)
    print("Encoded all")
    sys.stdout.flush()
    docids = [elem[0] for elem in new_data] 
    
    return (embeddings, docids)

def process_embeddings(embeddings, docids, out_docids, out_embeddings):
    numpy.save(out_docids, numpy.array(docids))

    embeddings_mat = numpy.asmatrix(embeddings)
    numpy.save(out_embeddings, embeddings_mat)

def main():
    if len(sys.argv) != 2:
        raise ValueError("Usage: python %s idx file-prefix\n" % sys.argv[0])
    
    prefix = sys.argv[1]
    model = "msmarco-distilbert-base-tas-b"
    embeddings, docids = compute_embeddings(model)
    docids_file = ("%s_docid.npy") % (prefix)
    embedding_file = ("%s_embeddings.npy") % (prefix)
    process_embeddings(embeddings, docids, docids_file, embedding_file)
    print(("Output to %s and %s") % (docids_file, embedding_file))

if __name__ == "__main__":
    main()