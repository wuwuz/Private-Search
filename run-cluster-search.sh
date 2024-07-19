#go run private-search.go -n 50000000 -d 128 -m 32 -k 10 -q 1000 -input ./SIFT-dataset/bigann_base.bvecs -query ./SIFT-dataset/bigann_query.bvecs \
#                         -output ./private-search-result.txt -report ./private-search-report.txt -gnd ./SIFT-dataset/gnd/idx_50M.ivecs \
#                         -step 30 -parallel 3 -rtt 50

python cluster-search.py -n 1000000 -d 128 -k 10 -q 1000 -input ./SIFT-dataset/bigann_base.bvecs -query ./SIFT-dataset/bigann_query.bvecs  \
                         -gnd ./SIFT-dataset/gnd/idx_1M.ivecs

python cluster-search.py -n 3201821 -d 192 -k 100 -q 1000 -input ./msmarco-dataset/msmarco_embeddings.npy  -query ./msmarco-dataset/msmarco_queries.npy \
                         -output ./msmarco-dataset/cluster_search_result.txt

#go run cluster-search.go -n 3201821 -d 192 -m 32 -k 100 -q 1000 -input ./msmarco-dataset/msmarco_embeddings.npy -query ./msmarco-dataset/msmarco_queries.npy \
#                         -report ./msmarco-dataset/report.txt \
#                         -step 15 -parallel 2 -rtt 50 