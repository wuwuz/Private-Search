package graphann

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"strconv"
	"testing"
	"time"

	"github.com/evan176/hnswgo"
)

func TestDistance(t *testing.T) {
	dim := 128

	v1 := make([]float32, dim)
	v2 := make([]float32, dim)

	for k := 0; k < 1000; k++ {

		for i := 0; i < dim; i++ {
			v1[i] = rand.Float32()
			v1[i] = rand.Float32()
		}

		truth := L2Dist(v1, v2)

		simdResult := L2DistSIMD(v1, v2)

		if math.Abs(float64(truth-simdResult)) >= 1e-4 {
			t.Fatalf("At the %d trial, L2Dist and L2DistSIMD do not match: %f != %f", k, truth, simdResult)
		}
	}

	// now we test the throughput of both versions

	rept := 1000000

	start := time.Now()
	for i := 0; i < rept; i++ {
		L2Dist(v1, v2)
	}
	end := time.Now()
	totalTime := end.Sub(start).Seconds()
	fmt.Println("L2Dist time: ", end.Sub(start))
	fmt.Println("Throughput: ", float64(rept)/totalTime, " per second")

	start = time.Now()
	for i := 0; i < rept; i++ {
		L2DistSIMD(v1, v2)
	}
	end = time.Now()
	totalTime = end.Sub(start).Seconds()
	fmt.Println("L2DistSIMD time: ", end.Sub(start))
	fmt.Println("Throughput: ", float64(rept)/totalTime, " per second")
}

func TestBuildGraphAndSearch(t *testing.T) {

	n := 1000000
	dim := 128
	m := 32

	savepath := "/home/mingxunz/Private-Search/SIFT-dataset/"
	dataset := "bigann_base"
	outputPrefix := dataset + "_" + strconv.Itoa(n) + "_" + strconv.Itoa(dim) + "_" + strconv.Itoa(m)
	fmt.Printf("Dataset: %s, Prefix :%s\n", dataset, outputPrefix)
	inputFile := savepath + "/" + dataset + ".bvecs"
	graphFile := savepath + outputPrefix + "_graph.npy"

	// Load vectors from file
	vectors, err := LoadBvecsFile(inputFile, n, dim)
	if err != nil {
		fmt.Printf("Failed to load vectors from file: %v\n", err)
		t.Fatal(err)
	} else {
		fmt.Println("Loaded vectors from file")
	}

	// now we build the graph
	var graph [][]int
	if _, err := os.Stat(graphFile); err == nil {
		fmt.Println("Graph file already exists, skipping building graph")
		graph, err = LoadGraphFromFile(graphFile, n, m)
		if err != nil {
			t.Fatal(err)
		}
	} else {
		fmt.Println("Building graph")
		graph = BuildGraph(n, dim, m, vectors, savepath, outputPrefix)
		SaveGraphToFile(graphFile, graph)
		fmt.Println("Graph built and saved to file")
	}

	fmt.Println("Graph size: ", len(graph), len(graph[0]))
	fmt.Println("Evaluating graph quality")
	EvaluateGraphQuality(vectors, graph)
}

func TestSearchQuality(t *testing.T) {

	n := 1000000
	dim := 128
	m := 32
	q := 1000

	savepath := "/home/mingxunz/Private-Search/SIFT-dataset/"
	dataset := "bigann_base"
	outputPrefix := dataset + "_" + strconv.Itoa(n) + "_" + strconv.Itoa(dim) + "_" + strconv.Itoa(m)
	fmt.Printf("Dataset: %s, Prefix :%s\n", dataset, outputPrefix)
	inputFile := savepath + "/" + dataset + ".bvecs"
	graphFile := savepath + outputPrefix + "_graph.npy"
	queryFile := savepath + "bigann_query.bvecs"
	gndFile := savepath + "gnd/idx_1M.ivecs"
	hnswFile := savepath + outputPrefix + "_hnsw.bin"

	// Load vectors from file
	vectors, err := LoadBvecsFile(inputFile, n, dim)
	if err != nil {
		fmt.Printf("Failed to load vectors from file: %v\n", err)
		t.Fatal(err)
	} else {
		fmt.Println("Loaded vectors from file")
	}

	// now we build the graph
	var graph [][]int
	if _, err := os.Stat(graphFile); err == nil {
		fmt.Println("Graph file already exists, skipping building graph")
		graph, err = LoadGraphFromFile(graphFile, n, m)
		if err != nil {
			t.Fatal(err)
		}
	} else {
		fmt.Println("Building graph")
		graph = BuildGraph(n, dim, m, vectors, savepath, outputPrefix)
		SaveGraphToFile(graphFile, graph)
		fmt.Println("Graph built and saved to file")
	}

	fmt.Println("Graph size: ", len(graph), len(graph[0]))

	g := BasicGraphInfo{
		N:       n,
		Dim:     dim,
		M:       m,
		Graph:   graph,
		Vectors: vectors,
	}

	frontend := GraphANNFrontend{
		Graph: &g,
	}

	frontend.Preprocess()

	// Load query vectors
	queryVectors, err := LoadBvecsFile(queryFile, q, dim)
	if err != nil {
		fmt.Printf("Failed to load query vectors from file: %v\n", err)
	}

	// Load ground truth
	gnd, err := LoadIvecsFile(gndFile, q, 100)
	if err != nil {
		fmt.Printf("Failed to load ground truth from file: %v\n", err)
	}

	// Search

	start := time.Now()
	answers, _ := frontend.SearchKNNBatch(queryVectors, 100, 20, 2, false)
	end := time.Now()

	// evaluate recall
	recall := ComputeRecall(gnd, answers, 10)

	fmt.Println("Recall: ", recall)
	fmt.Println("Average search time: ", end.Sub(start).Seconds()/float64(q), " seconds per query")

	h := hnswgo.New(dim, m, 300, 100, uint32(n), "l2")

	// only when the hnsw index file does not exist, we create a new hnsw index
	if _, err := os.Stat(hnswFile); os.IsNotExist(err) {
		fmt.Println("Creating HNSW index")
		start := time.Now()
		for i := 0; i < n; i++ {
			h.AddPoint(vectors[i], uint32(i))
		}
		end := time.Now()
		h.Save(hnswFile)
		fmt.Println("HNSW index created, time = ", end.Sub(start))
	} else {
		h = hnswgo.Load(hnswFile, dim, "l2")
		fmt.Println("Loaded HNSW index from file")
	}

	hnswAnswer := make([][]int, q)
	// search using hnsw
	for i := 0; i < q; i++ {
		ans, _ := h.SearchKNN(queryVectors[i], 100)
		hnswAnswer[i] = make([]int, len(ans))
		for j, v := range ans {
			hnswAnswer[i][j] = int(v)
		}
	}

	// evaluate recall
	hnswRecall := ComputeRecall(gnd, hnswAnswer, 10)
	fmt.Println("HNSW Recall: ", hnswRecall)
}
