#!/bin/bash
# define variables
n=2000000 #3201821 # number of vectors
d=192 # dimension of vectors
m=32 # number of neighbors in the graph
k=100 # number of outputs to retrieve
q=100 # number of queries 
step=15 # max number of steps in the graph
parallel=2 # number of parallel queries to make in one step
rtt=50 # round trip time in milliseconds
report="synthetic_report.txt"
# if true, skip the preprocessing and just run the online
skip="true"
flag=""

# if skip == true, add --skip to the flag
if [ "$skip" == "true" ]; then
    flag="--skip"
fi

# run the backend
go run backend/backend.go --synthetic -n $n -d $d -m $m $flag &
# run the frontend
go run frontend/frontend.go -n $n -d $d -m $m -k $k -q $q -step $step -parallel $parallel -rtt $rtt -report $report $flag


# Get the process ID of "backend"
pid=$(pgrep backend)

# terminate the backend
if [ -z "$pid" ]; then
    echo "No process named 'backend' is currently running."
else
    echo "The process ID of 'backend' is: $pid"
    kill -9 $pid
fi