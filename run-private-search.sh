go run private-search.go -n 100000000 -d 128 -m 32 -k 10 -q 100 \
                         -step 20 -parallel 2 -rtt 50 -input "synthetic"

#go run private-search.go -n 1000000 -d 128 -m 32 -k 10 -q 1000 -input ./SIFT-dataset/bigann_base.bvecs -query ./SIFT-dataset/bigann_query.bvecs \
#                         -output ./private-search-result.txt -report ./private-search-report.txt -gnd ./SIFT-dataset/gnd/idx_1M.ivecs \
#                         -step 20 -parallel 2 -rtt 50 -nonprivate


#go run private-search.go -n 3201821 -d 192 -m 32 -k 100 -q 1000 -input ./msmarco-dataset/msmarco_embeddings.npy -query ./msmarco-dataset/msmarco_queries.npy \ 
#                         -report ./msmarco-dataset/report.txt \
                         #-step 15 -parallel 2 -rtt 50 
#go run private-search.go -n 1000000 -d 128 -m 32 -k 10 -q 1000 -input ./SIFT-dataset/bigann_base.bvecs -query ./SIFT-dataset/bigann_query.bvecs \
#                         -output ./private-search-result.txt -report ./private-search-report.txt -gnd ./SIFT-dataset/gnd/idx_1M.ivecs \
#                         -step 15 -parallel 2 -rtt 50