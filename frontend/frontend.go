package main

import (
	"container/heap"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"net/http"
	"os"
	"sort"
	"time"
)

const ServerAddress = "localhost:8080"

var n int
var d int
var m int
var q int
var outputK int
var skipPrep bool
var queryVectors [][]float32
var maxStep int
var parallel int
var rtt int

func genRandomVector(dim int) []float32 {
	ret := make([]float32, dim)
	for i := 0; i < dim; i++ {
		ret[i] = rand.Float32()
	}
	return ret
}

func genRandomQueryVectors(num int, dim int) [][]float32 {
	ret := make([][]float32, num)
	for i := 0; i < num; i++ {
		ret[i] = genRandomVector(dim)
	}
	return ret
}

func readQueryFromFile(filename string) ([][]float32, error) {
	// open the file as read only
	log.Printf("shape of the query vectors: %d %d\n", q, d)
	ret := genRandomQueryVectors(q, d) // default to random vectors
	file, err := os.Open(filename)
	if err != nil {
		panic(err)
	}
	for i := 0; i < q; i++ {
		// read a line of d float values
		for j := 0; j < d; j++ {
			_, err := fmt.Fscanf(file, "%f", &ret[i][j])
			if err != nil {
				return nil, err
			}
		}
	}
	return ret, nil
}

type vertexInfo struct {
	index     int
	neighbors []int
	vector    []float32
}

var fastStartVertices map[int]vertexInfo

type jsonResponse struct {
	Neighbors []int
	MatrixRow []float32
}

func MakeGraphQuery(indices []int, serverAddress string) ([]vertexInfo, error) {
	// make a batch of queries to the server
	// use http query

	// the python euivalent code is
	//     query_string = '&'.join([f'rowIndex={idx}' for idx in to_query_list])
	//     response = requests.get(f'http://localhost:8080/query?{query_string}')

	query_string := ""
	for _, idx := range indices {
		query_string += fmt.Sprintf("&q=%d", idx)
	}

	query_string = "http://" + serverAddress + "/query?" + query_string

	// send the query to the server
	// and get the response as a json object

	resp, err := http.Get(query_string)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	// now we need to parse the json object
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	var data []jsonResponse
	err = json.Unmarshal(body, &data)
	if err != nil {
		return nil, err
	}

	ret := make([]vertexInfo, len(indices))
	for i := 0; i < len(indices); i++ {
		ret[i] = vertexInfo{
			index:     indices[i],
			neighbors: data[i].Neighbors,
			vector:    data[i].MatrixRow,
		}
	}

	return ret, nil
}

func calcDist(a []float32, b []float32) float32 {
	dist := float32(0.0)
	for i := 0; i < len(a); i++ {
		dist += (a[i] - b[i]) * (a[i] - b[i])
	}
	return float32(dist)
}

func MakeNonPrivateQuery(serverAddress string, indices []int) ([]vertexInfo, error) {
	// make a batch of queries to the server
	// use http query

	// the python euivalent code is
	//     query_string = '&'.join([f'rowIndex={idx}' for idx in to_query_list])
	//     response = requests.get(f'http://localhost:8080/query?{query_string}')

	query_string := ""
	for _, idx := range indices {
		query_string += fmt.Sprintf("&q=%d", idx)
	}

	query_string = "http://" + serverAddress + "/nonprivatequery?" + query_string

	// send the query to the server
	// and get the response as a json object

	resp, err := http.Get(query_string)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	// now we need to parse the json object
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		panic(err)
	}

	var data []jsonResponse
	err = json.Unmarshal(body, &data)
	if err != nil {
		panic(err)
	}

	ret := make([]vertexInfo, len(indices))
	for i := 0; i < len(indices); i++ {
		ret[i] = vertexInfo{
			index:     indices[i],
			neighbors: data[i].Neighbors,
			vector:    data[i].MatrixRow,
		}
	}

	return ret, nil
}

type toBeExploredItem struct {
	dist float32
	id   int
}

type exploreQueue []*toBeExploredItem

func (pq exploreQueue) Len() int { return len(pq) }
func (pq exploreQueue) Less(i, j int) bool {
	return pq[i].dist < pq[j].dist
}
func (pq exploreQueue) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
}
func (pq *exploreQueue) Push(x interface{}) {
	item := x.(*toBeExploredItem)
	*pq = append(*pq, item)
}
func (pq *exploreQueue) Pop() interface{} {
	old := *pq
	n := len(old)
	item := old[n-1]
	*pq = old[0 : n-1]
	return item
}

func MakeANNQuery(serverAddress string, queryVector []float32, k int, maxStep int, parallel int, benchmarking bool) []int {
	// in our python implementation, we will store extra vertex info to fast-start the first round
	// here, we just do the normal starting by starting from vertex 0...parallel - 1
	//log.Printf("Starting ANN query with %d vertices, %d neighbors, %d queries, %d output, %d steps, %d parallel, %t skip prep\n",
	//	n, m, q, k, maxStep, parallel, skipPrep)

	knownVertices := map[int]vertexInfo{}
	// define a priority queue
	// the priority queue is a min heap, ranked by the distance to the query vector
	// each element is a tuple (*potential distance, vertex index)
	// we first push the first parallel * m vertices into the heap
	toBeExploredItems := make(exploreQueue, 0)
	heap.Init(&toBeExploredItems)

	// we first find the top parallel vertices from fastStartVertices by their distance to the query vector
	// if we are benchmarking, we skip this step
	if !benchmarking {
		fastStartQueue := make(exploreQueue, 0)
		for _, v := range fastStartVertices {
			dist := calcDist(v.vector, queryVector)
			fastStartQueue.Push(&toBeExploredItem{dist: dist, id: v.index})
		}
		sort.Sort(fastStartQueue)
		for i := 0; len(toBeExploredItems) < parallel; i++ {
			v := fastStartQueue[i]
			if _, ok := knownVertices[v.id]; ok {
				// we have already known this vertex
				continue
			}
			knownVertices[v.id] = fastStartVertices[v.id]
			heap.Push(&toBeExploredItems, v)
			//toBeExploredItems = append(toBeExploredItems, v)
		}
	}

	for step := 0; step < maxStep; step++ {

		// each time we issue parallel batches, each exploring one vertex's neighbors
		batchQ := make([]int, 0, m)
		for rept := 0; rept < parallel; rept++ {
			if len(toBeExploredItems) == 0 || benchmarking {
				// in this case we simply make random queries
				for i := 0; i < m; i++ {
					batchQ = append(batchQ, rand.Intn(n))
				}
			} else {
				item := heap.Pop(&toBeExploredItems).(*toBeExploredItem)
				v := item.id
				//log.Print("Exploring vertex ", v, " at step ", step, " with distance ", item.dist)
				// copy the neighbors of v to the batchQ
				batchQ = append(batchQ, knownVertices[v].neighbors...)
			}

			//log.Printf("Exploring %d vertices at step %d", len(batchQ), step)
			//log.Printf("The first 5 vertices are %v", batchQ[:5])

		}

		queryResults, err := MakeGraphQuery(batchQ, serverAddress)
		if err != nil {
			panic(err)
		}

		if benchmarking {
			// if we are just benchmarking, we don't care about the return
			continue
		}

		for _, v := range queryResults {
			if _, ok := knownVertices[v.index]; ok {
				// we have already known this vertex
				continue
			}
			// if the neighbor list is all zeroes, we skip this vertex
			ok := false
			for _, neighbor := range v.neighbors {
				if neighbor != 0 {
					ok = true
					break
				}
			}
			if ok {
				knownVertices[v.index] = v
				// calculate the distance to the query vector
				dist := calcDist(v.vector, queryVector)
				heap.Push(&toBeExploredItems, &toBeExploredItem{dist: dist, id: v.index})
			}
		}
	}

	// extract all known vertices and sort them by distance by ascending order
	allKnownVertices := make([]toBeExploredItem, 0, len(knownVertices))
	for _, v := range knownVertices {
		allKnownVertices = append(allKnownVertices,
			toBeExploredItem{
				dist: calcDist(v.vector, queryVector),
				id:   v.index,
			})
	}
	sort.Slice(allKnownVertices, func(i, j int) bool {
		return allKnownVertices[i].dist < allKnownVertices[j].dist
	})
	ret := make([]int, k)
	for i := 0; i < k; i++ {
		if i >= len(allKnownVertices) {
			ret[i] = -1
		} else {
			ret[i] = allKnownVertices[i].id
		}
	}
	return ret
}

func generateReport(filename string, serverAddress string, totalTime time.Duration, avgTime float64) {
	// only append the report to the file
	file, err := os.OpenFile(filename, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		panic(err)
	}

	query_string := "http://" + serverAddress + "/info"

	// send the query to the server
	// and get the response as a json object

	resp, err := http.Get(query_string)
	if err != nil {
		panic(err)
	}
	defer resp.Body.Close()

	// now we need to parse the json object
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		panic(err)
	}

	var data map[string]interface{}
	err = json.Unmarshal(body, &data)
	if err != nil {
		panic(err)
	}

	DBSize := data["DBSize"].(float64) // in bytes
	log.Print("DB Size: ", DBSize)
	PrepTime := data["PrepTime"].(float64)
	log.Printf("Prep Time: %f", PrepTime)
	Storage := data["Storage"].(float64) // in bytes
	log.Printf("Storage: %f", Storage)
	OnlineComm := data["OnlineComm"].(float64) // in bytes
	log.Printf("Online Comm: %f", OnlineComm)
	OfflineComm := data["OfflineComm"].(float64) // in bytes
	log.Printf("Offline Comm: %f", OfflineComm)

	fmt.Fprintf(file, "-------------------------\n")
	fmt.Fprintf(file, "Private ANN Benchmarking w/ Go Frontend\n")
	fmt.Fprintf(file, "Settings:\n")
	fmt.Fprintf(file, "** Vector Num: %d\n", n)
	fmt.Fprintf(file, "** DB Size (MB): %f\n", float64(DBSize)/1024.0/1024.0)
	fmt.Fprintf(file, "** Top K: %d\n", outputK)
	fmt.Fprintf(file, "** Rounds: %d\n", maxStep)
	fmt.Fprintf(file, "** Parallel Exploration: %d\n", parallel)
	fmt.Fprintf(file, "** RTT (ms): %d\n", rtt)
	fmt.Fprintf(file, "\n")
	fmt.Fprintf(file, "Preprocessing Cost:\n")
	fmt.Fprintf(file, "** Storage (MB): %f\n", float64(Storage)/1024.0/1024.0)
	fmt.Fprintf(file, "** Preparation Time (s): %f\n", PrepTime)
	fmt.Fprintf(file, "** Offline Communication Cost Per Q (KB, amt.): %f\n", float64(OfflineComm)*float64(maxStep)*float64(parallel)/1024.0)
	fmt.Fprintf(file, "\n")
	fmt.Fprintf(file, "Online Cost:\n")
	fmt.Fprintf(file, "** Average Computation Time Per Query (s): %f\n", avgTime)
	fmt.Fprintf(file, "** Average Total Time Per Q (s): %f\n", avgTime+float64(rtt)/1000.0*float64(maxStep))
	fmt.Fprintf(file, "** Online Communication Per Q (KB): %f\n", float64(OnlineComm)*float64(maxStep)*float64(parallel)/1024.0)
	fmt.Fprintf(file, "-----------------------\n")
}

func main() {

	// Parameters
	// "-n": number of vectors
	// "-d": dimension of the vectors
	// "-m": number of neighbors
	// "-q": number of queries
	// "-k": top K output
	// "-input": input file name, default to synthetic
	// "-output": output file name, default to null
	// "--skip": skipping preprocessing. Only testing the query time
	// "--step": searching max depth, default to 15
	// "--parallel": how many parallel vertices are accessed in the same round, default to 1

	numVectors := flag.Int("n", 3201821, "number of vectors")
	dimVectors := flag.Int("d", 192, "dimension of the vectors")
	neighborNum := flag.Int("m", 32, "number of neighbors")
	outputNum := flag.Int("k", 100, "top K output")
	queryNum := flag.Int("q", 100, "number of queries")
	queryFile := flag.String("file", "synthetic", "file name")
	outputFile := flag.String("output", "", "output file name")
	reportFile := flag.String("report", "", "report file name")
	stepN := flag.Int("step", 15, "searching max depth")
	parallelN := flag.Int("parallel", 1, "how many parallel vertices are accessed in the same round")
	skip := flag.Bool("skip", false, "skip preprocessing")
	rttN := flag.Int("rtt", 50, "round trip time in ms")

	flag.Parse()

	n = *numVectors
	d = *dimVectors
	m = *neighborNum
	skipPrep = *skip
	q = *queryNum
	outputK = *outputNum
	maxStep = *stepN
	parallel = *parallelN
	rtt = *rttN

	if *queryFile == "synthetic" {
		queryVectors = genRandomQueryVectors(q, d)
	} else {
		qv, err := readQueryFromFile(*queryFile)
		if err != nil {
			panic(err)
		}
		queryVectors = qv
		log.Printf("Read %d query vectors from file %s\n", q, *queryFile)
		log.Printf("The first vector is %v\n", queryVectors[0][:5])
	}

	// polling the server
	for {
		resp, err := http.Get("http://localhost:8080/info")
		if err != nil {
			//fmt.Println("Backend is not up yet, retrying in 3 seconds...")
			time.Sleep(3 * time.Second)
			continue
		}
		resp.Body.Close()

		if resp.StatusCode == http.StatusOK {
			fmt.Println("Backend is up!")
			break
		} else {
			//fmt.Println("Backend is not ready yet, retrying in 3 seconds...")
			time.Sleep(3 * time.Second)
		}
	}

	// fast starting
	// build up the fast starting vertices by just randomly querying the server
	targetFastStartVertices := int(math.Sqrt(float64(n)))
	log.Printf("Fast starting with %d vertices\n", targetFastStartVertices)
	fastStartVertices = make(map[int]vertexInfo)

	start := time.Now()

	tmp := 0
	for i := 0; i < targetFastStartVertices/m; i++ {
		tmp += 1
		batchQ := make([]int, 0, m)
		for j := 0; j < m; j++ {
			batchQ = append(batchQ, rand.Intn(n))
		}
		results, err := MakeNonPrivateQuery(ServerAddress, batchQ)
		if err != nil {
			panic(err)
		}
		//fastStartVertices = append(fastStartVertices, results...)
		for _, v := range results {
			fastStartVertices[v.index] = v
		}
	}

	end := time.Now()
	fmt.Println("Average non-private batch query time: ", end.Sub(start).Seconds()/float64(tmp))

	// we now make 100 private queries to the server and get the average time

	start = time.Now()
	for i := 0; i < 100; i++ {
		batchQ := make([]int, 0, m)
		for j := 0; j < m; j++ {
			batchQ = append(batchQ, rand.Intn(n))
		}

		_, err := MakeGraphQuery(batchQ, ServerAddress)
		if err != nil {
			panic(err)
		}
	}
	end = time.Now()
	fmt.Println("Average private batch query time: ", end.Sub(start).Seconds()/100.0)

	// we make 1000 info queries to the server and get the average time

	start = time.Now()
	for i := 0; i < 1000; i++ {
		resp, err := http.Get("http://localhost:8080/info")
		if err != nil {
			panic(err)
		}
		resp.Body.Close()
	}
	end = time.Now()
	fmt.Println("Average info query time: ", end.Sub(start).Seconds()/1000.0)

	result := make([][]int, q)
	start = time.Now()

	for i := 0; i < q; i++ {
		ans := MakeANNQuery(ServerAddress, queryVectors[i], outputK, maxStep, parallel, skipPrep)
		result[i] = ans
	}

	end = time.Now()
	totalTime := end.Sub(start)
	fmt.Println("Total time: ", totalTime)
	avgTime := totalTime.Seconds() / float64(q)
	fmt.Println("Average time: ", avgTime)
	// compute the average time plus the round trip time * step
	fmt.Println("Average time with RTT: ", avgTime+float64(rtt)/1000.0*float64(maxStep))

	if *outputFile != "" {
		// open the file as write only and rewrite mode
		// if the file does not exist, create it
		file, err := os.Create(*outputFile)
		if err != nil {
			panic(err)
		}
		//then output the results to the file
		for i := 0; i < q; i++ {
			for j := 0; j < outputK; j++ {
				fmt.Fprintf(file, "%d ", result[i][j])
			}
			fmt.Fprintf(file, "\n")
		}
	}

	if *reportFile != "" {
		generateReport(*reportFile, ServerAddress, totalTime, avgTime)
	}
}
