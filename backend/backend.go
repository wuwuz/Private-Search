package main

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"net/http"
	"os"
	"strconv"
	"strings"

	"example.com/private-search/pianopir"
)

var matrix [][]float32
var graph [][]uint32
var rawDB []uint64
var DBSize uint64
var DBEntryByteNum uint64
var vectorSize int
var numNeighbors int
var PIR *pianopir.SimpleBatchPianoPIR
var skipPrep bool

// embeddings file name
const embeddingsFileDefault = "msmarco_embeddings_reduced_permuted.txt"
const graphFileDefault = "graph_permuted.txt"

//const embeddingsFile = "embeddings100.txt"
//const graphFile = "graph100.txt"

func loadMatrix(filename string) error {
	file, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		strVals := strings.Split(line, " ")
		var row []float32
		for _, strVal := range strVals {
			val, err := strconv.ParseFloat(strVal, 32)
			if err != nil {
				return err
			}
			row = append(row, float32(val))
		}
		matrix = append(matrix, row)
	}

	// output the shape of the matrix
	fmt.Println("Matrix shape: ", len(matrix), len(matrix[0]))

	return scanner.Err()
}

func genRandomMatrix(num int, dim int) {
	for i := 0; i < num; i++ {
		var row []float32
		for j := 0; j < dim; j++ {
			row = append(row, rand.Float32())
		}
		matrix = append(matrix, row)
	}

	// output the shape of the matrix
	fmt.Println("Matrix shape: ", len(matrix), len(matrix[0]))
}

func loadGraph(filename string) error {
	file, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		strVals := strings.Split(line, " ")
		var row []uint32
		for _, strVal := range strVals {
			val, err := strconv.Atoi(strVal)
			if err != nil {
				return err
			}
			row = append(row, uint32(val))
		}
		graph = append(graph, row)
	}

	//output the shape of the graph
	fmt.Println("Graph shape: ", len(graph), len(graph[0]))
	return scanner.Err()
}

func genRandomGraph(num int, neigh int) {
	for i := 0; i < num; i++ {
		var row []uint32
		for j := 0; j < neigh; j++ {
			k := rand.Intn(num)
			for k == i {
				// no self loop
				k = rand.Intn(num)
			}
			row = append(row, uint32(k))
		}
		graph = append(graph, row)
	}

	//output the shape of the graph
	fmt.Println("Graph shape: ", len(graph), len(graph[0]))
}

// makeRawDB converts the matrix and graph into a rawDB
// we concatenate each row of the matrix and also the correspoding neighbor lists
// and we
func MakeRawDB(matrix [][]float32, graph [][]uint32) (uint64, uint64, []uint64) {
	// let's first compute the DBEntryByteNum
	vectorSize := uint64(len(matrix[0]))
	numNeighbors := uint64(len(graph[0]))

	DBEntryByteNum := vectorSize*4 + numNeighbors*4
	DBSize := uint64(len(matrix))

	fmt.Println("DBEntryByteNum: ", DBEntryByteNum)
	fmt.Println("DBSize: ", DBSize)

	rawDB := make([]uint64, DBSize*DBEntryByteNum/8)

	for i := uint64(0); i < DBSize; i++ {
		// we first convert the matrix row to a byte slice
		matrixRow := matrix[i]
		matrixRowBytes := make([]byte, vectorSize*4)
		for j := uint64(0); j < vectorSize; j++ {
			binary.LittleEndian.PutUint32(matrixRowBytes[j*4:], math.Float32bits(matrixRow[j]))
		}

		// we also convert the graph row to a byte slice
		neighbors := graph[i]
		neighborsBytes := make([]byte, numNeighbors*4)
		for j := uint64(0); j < numNeighbors; j++ {
			binary.LittleEndian.PutUint32(neighborsBytes[j*4:], neighbors[j])
		}

		// then we concatenate the two byte slices
		entryBytes := append(matrixRowBytes, neighborsBytes...)

		// then we convert the byte slice to a uint64 slice
		entry := make([]uint64, DBEntryByteNum/8)
		for j := uint64(0); j < DBEntryByteNum/8; j++ {
			entry[j] = binary.LittleEndian.Uint64(entryBytes[j*8:])
		}

		// we then copy the entry to the rawDB
		copy(rawDB[i*DBEntryByteNum/8:], entry)
	}

	return uint64(DBSize), uint64(DBEntryByteNum), rawDB
}

func ConvertFromRawDB(vectorSize int, numNeighbors int, entry []uint64) ([]float32, []uint32) {
	// we first convert the entry to a byte slice
	entryBytes := make([]byte, len(entry)*8)
	for i := 0; i < len(entry); i++ {
		binary.LittleEndian.PutUint64(entryBytes[i*8:], entry[i])
	}

	// for the first vectorSize*4 bytes, we convert it to a float32 slice
	vector := make([]float32, vectorSize)
	for i := 0; i < vectorSize; i++ {
		vector[i] = math.Float32frombits(binary.LittleEndian.Uint32(entryBytes[i*4:]))
	}

	// for the next numNeighbors*4 bytes, we convert it to a uint32 slice
	neighbors := make([]uint32, numNeighbors)
	for i := 0; i < numNeighbors; i++ {
		neighbors[i] = binary.LittleEndian.Uint32(entryBytes[(vectorSize+i)*4:])
	}

	return vector, neighbors
}

func nonPrivateQueryHandler(w http.ResponseWriter, r *http.Request) {
	// TODO: implement non-private query handler
	query := r.URL.Query()
	rowIndexesStr := query["rowIndex"]

	//fmt.Println("Querying for: ", rowIndexesStr)

	// first we make a list storing all the indices
	var indices []uint64
	for _, rowIndexStr := range rowIndexesStr {
		rowIndex, err := strconv.Atoi(rowIndexStr)
		if err != nil || rowIndex < 0 || rowIndex >= len(matrix) {
			http.Error(w, "Row index out of range or invalid", http.StatusBadRequest)
			return
		}
		indices = append(indices, uint64(rowIndex))
	}

	vectors := make([][]float32, len(indices))
	neighbors := make([][]uint32, len(indices))

	for i := 0; i < len(indices); i++ {
		vectors[i] = matrix[indices[i]]
		neighbors[i] = graph[indices[i]]
	}
	var responseMap []map[string]interface{}
	for i := 0; i < len(indices); i++ {
		responseData := map[string]interface{}{
			"matrixRow": vectors[i],
			"neighbors": neighbors[i],
		}
		responseMap = append(responseMap, responseData)

		if skipPrep {
			// in this case we don't care about the correctness
			continue
		}
	}

	jsonResponse, err := json.Marshal(responseMap)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.Write(jsonResponse)
}

func queryHandler(w http.ResponseWriter, r *http.Request) {
	query := r.URL.Query()
	rowIndexesStr := query["rowIndex"]

	//fmt.Println("Querying for: ", rowIndexesStr)

	// first we make a list storing all the indices
	var indices []uint64
	for _, rowIndexStr := range rowIndexesStr {
		rowIndex, err := strconv.Atoi(rowIndexStr)
		if err != nil || rowIndex < 0 || rowIndex >= len(matrix) {
			http.Error(w, "Row index out of range or invalid", http.StatusBadRequest)
			return
		}
		indices = append(indices, uint64(rowIndex))
	}

	// then we make a batch query
	//fmt.Println("Querying for: ", indices)
	responses, _ := PIR.Query(indices)
	//fmt.Println("Got responses: ", responses)

	vectors := make([][]float32, len(responses))
	neighbors := make([][]uint32, len(responses))

	// convert the responses to vectors and neighbors
	for i, response := range responses {
		vectors[i], neighbors[i] = ConvertFromRawDB(vectorSize, numNeighbors, response)
	}

	var responseMap []map[string]interface{}
	for i := 0; i < len(responses); i++ {
		responseData := map[string]interface{}{
			"matrixRow": vectors[i],
			"neighbors": neighbors[i],
		}
		responseMap = append(responseMap, responseData)

		if skipPrep {
			// in this case we don't care about the correctness
			continue
		}

		// verify the neighbors are correct
		// there are two cases: either the neighbors are all zeros, or they match the original graph
		allZeros := true
		allMatch := true
		for j := 0; j < len(neighbors[i]); j++ {
			if neighbors[i][j] != 0 {
				allZeros = false
			}
			if neighbors[i][j] != graph[indices[i]][j] {
				allMatch = false
			}
		}

		if !allZeros && !allMatch {
			// generate an error
			fmt.Println("Error: neighbors do not match the original graph")
		}
	}

	jsonResponse, err := json.Marshal(responseMap)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.Write(jsonResponse)
}

// infoQueryHandler is a handler for the /info endpoint
// when the client sends a GET request to /info, the server should return the following information in JSON format:
// Database size,
// Preprocessing computatinon time (per batch),
// Local storage size,
// Communication Cost (online per batch),
// Communication Cost (maintenance per batch),
func infoQueryHandler(w http.ResponseWriter, r *http.Request) {

	info := map[string]interface{}{
		"DBSize":      DBSize * DBEntryByteNum,
		"PrepTime":    PIR.PreprocessingTime(),
		"Storage":     PIR.LocalStorageSize(),
		"OnlineComm":  PIR.CommCostPerBatchOnline(),
		"OfflineComm": PIR.CommCostPerBatchOffline(),
	}

	jsonResponse, err := json.Marshal(info)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.Write(jsonResponse)

}

func main() {

	// If the file is called with argument --synthetic, then we will run the synthetic test
	// If the file is called with argument --real, then we will run the real test
	syntheticTest := flag.Bool("synthetic", false, "run synthetic test")
	realTest := flag.Bool("real", false, "run real test")
	skipPreprocessing := flag.Bool("skip", false, "skip preprocessing")
	// For real testing:
	// Add an input parameter --embfile to specify the embeddings file, with a default value
	// Add an input parameter --graphfile to specify the graph file, with a default value
	embeddingsFile := flag.String("embfile", embeddingsFileDefault, "embeddings file")
	graphFile := flag.String("graphfile", graphFileDefault, "graph file")
	// For synthetic testing:
	// Add an input parameter -n to specify the number of vectors, default to 1000000
	// Add an input parameter -d to specify the dimension of the vectors, default to 192
	// Add an input parameter -m to specify the number of neighbors, default to 32
	num := flag.Int("n", 1000000, "number of vectors")
	dim := flag.Int("d", 192, "dimension of the vectors")
	neigh := flag.Int("m", 32, "number of neighbors")

	flag.Parse()

	skipPrep = *skipPreprocessing

	if *realTest {
		log.Printf("Loading embeddings from %v and graph from %v\n", *embeddingsFile, *graphFile)
		err := loadMatrix(*embeddingsFile)
		if err != nil {
			panic(err)
		}
		err = loadGraph(*graphFile)
		if err != nil {
			panic(err)
		}
	}

	if *syntheticTest {
		log.Printf("Generating random matrix and graph with %v vectors, %v dimensions, %v neighbors\n", *num, *dim, *neigh)
		genRandomMatrix(*num, *dim)
		genRandomGraph(*num, *neigh)
	}

	if len(matrix) == 0 || len(graph) == 0 {
		panic("No matrix or graph loaded")
	}

	vectorSize = len(matrix[0])
	numNeighbors = len(graph[0])

	DBSize, DBEntryByteNum, rawDB = MakeRawDB(matrix, graph)
	PIR = pianopir.NewSimpleBatchPianoPIR(DBSize, DBEntryByteNum, uint64(len(graph[0])), rawDB, 8)

	if skipPrep {
		PIR.DummyPreprocessing()
	} else {
		PIR.Preprocessing()
	}
	log.Printf("PIR config: %v\n", PIR.Config())
	log.Printf("PIR local storage size: %v MB\n", PIR.LocalStorageSize()/1024/1024)

	http.HandleFunc("/query", queryHandler)
	http.HandleFunc("/info", infoQueryHandler)
	http.HandleFunc("/nonprivatequery", nonPrivateQueryHandler)
	fmt.Println("Server started on :8080")
	http.ListenAndServe(":8080", nil)
}
