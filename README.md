Files
`msmarco-prefix.tsv`: includes all the documents and their first 512 characters (3M docs)
`msmarco-docdev-queries.tsv`: include 5k queries
`msmarco-queries-1000.tsv`: include 1k queries
`msmarco-docdev-qrels.tsv`: include the 5k queries' ground truths

`embeddings_msmarco.py`: generate the embeddings based on the prefix file
-- `msmarco_embeddings.npy`: the embedding numpy file (3M vectors, 768-dim)
-- `msmarco_docid.npy`: the docid for each vector

`dim-reduce.py`: Dimension reduction based on the embedding file
-- `ipca_msmarco.pkl`: the dim reduction model, can used elsewhere
-- `msmarco_embeddings_reduced.npy`: the dim reduction result numpy file (3M vectors, 192-dim)

`cluster-msmarco.py`: Clusters the embeddings
-- `msmarco-1280-cluster-labels.npy`: The cluster label for each vector (3M labels, 1280 clusters)
-- `msmarco-1280-cluster-centroids.npy`: The centroids of each cluster
-- `msmarco-1280-cluster-reps.npy`: The vector IDs that are closest to the centroids

`search-faiss-msmarco.py`: Do search based on HNSW vector search based on the FAISS package.
-- `faiss-msmarco-reduced-results.txt`: The results

`search-cluster-msmarco.py`: Do search based on the Tiptoe algorithm. First find the closest cluster, then search in the cluster.
-- `cluster-msmarco-reduced-results.txt`: The results

`search-cluster-msmarco-PIANO.py`: Do search modified to the PIANO PIR algorithm. Each cluster is a database document of size equal to the cluster size.
-- `cluster-msmarco-PIANO-results.txt`: PIANO Results
-- `cluster-msmarco-TIPTOE-results.txt`: Tiptoe Results (For comparison)
`mrr.py`: Do search quality evaluation. Run `python mrr.py ResultFilePath`. 
-- "RANKED" means how many ground truth are hit.
-- "MRR" is a quality metric of the search. It can be interpreted as $1/(avg hit rank)$. So if you always find the hit in the first result, $MRR=1$. If you always find the hit in the 10-th result, $MRR=0.1$.