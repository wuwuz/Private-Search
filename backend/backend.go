package main

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"log"
	"math"
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

// embeddings file name
const embeddingsFile = "msmarco_embeddings_reduced_permuted.txt"
const graphFile = "graph_permuted.txt"

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

func main() {

	err := loadMatrix(embeddingsFile)
	if err != nil {
		panic(err)
	}
	err = loadGraph(graphFile)
	if err != nil {
		panic(err)
	}

	vectorSize = len(matrix[0])
	numNeighbors = len(graph[0])

	DBSize, DBEntryByteNum, rawDB = MakeRawDB(matrix, graph)
	PIR = pianopir.NewSimpleBatchPianoPIR(DBSize, DBEntryByteNum, uint64(len(graph[0])), rawDB, 8)
	PIR.Preprocessing()
	log.Printf("PIR config: %v\n", PIR.Config())
	log.Printf("PIR local storage size: %v MB\n", PIR.LocalStorageSize()/1024/1024)

	http.HandleFunc("/query", queryHandler)
	fmt.Println("Server started on :8080")
	http.ListenAndServe(":8080", nil)
}
