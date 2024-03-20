package pianopir

import (
	"fmt"
	//"encoding/binary"

	"log"
	"math"
	"math/rand"
	"time"
)

const (
	//FailureProbLog2     = 40
	DefaultProgramPoint = 0x7fffffff
)

type PianoPIRConfig struct {
	DBEntryByteNum  uint32 // the number of bytes in a DB entry
	DBEntrySize     uint32 // the number of uint64 in a DB entry
	DBSize          uint32
	ChunkSize       uint32
	SetSize         uint32
	ThreadNum       uint32
	FailureProbLog2 uint32
}

type PianoPIRServer struct {
	config *PianoPIRConfig
	rawDB  []uint64
}

// an initialization function for the server
func NewPianoPIRServer(config *PianoPIRConfig, rawDB []uint64) *PianoPIRServer {
	return &PianoPIRServer{
		config: config,
		rawDB:  rawDB,
	}
}

func (s *PianoPIRServer) NonePrivateQuery(idx uint32) ([]uint64, error) {
	ret := make([]uint64, s.config.DBEntrySize)
	// initialize ret to be all zeros
	for i := uint32(0); i < s.config.DBEntrySize; i++ {
		ret[i] = 0
	}

	if idx >= s.config.DBSize {
		//log.Fatalf("idx %v is out of range", idx)
		if idx < s.config.ChunkSize*s.config.SetSize {
			// caused by the padding
			return ret, nil
		} else {
			// return an empty entry and an error
			return ret, fmt.Errorf("idx %v is out of range", idx)
		}
	}

	// copy the idx*DBEntrySize-th to (idx+1)*DBEntrySize-th elements to ret
	copy(ret, s.rawDB[idx*s.config.DBEntrySize:(idx+1)*s.config.DBEntrySize])
	return ret, nil
}

// the private query just computes the xor sum of the elements in the idxs list
func (s *PianoPIRServer) PrivateQuery(idxs []uint32) ([]uint64, error) {
	ret := make([]uint64, s.config.DBEntrySize)
	// initialize ret to be all zeros
	for i := uint32(0); i < s.config.DBEntrySize; i++ {
		ret[i] = 0
	}

	for _, idx := range idxs {
		// if idx is bigger than the rawDB size, just ignore
		if idx >= s.config.DBSize {
			continue
		}

		// xor the idx*DBEntrySize-th to (idx+1)*DBEntrySize-th elements to ret
		EntryXor(ret, s.rawDB[idx*s.config.DBEntrySize:(idx+1)*s.config.DBEntrySize], s.config.DBEntrySize)

		//for i := uint32(0); i < s.config.DBEntrySize; i++ {
		//	ret[i] ^= s.rawDB[idx*s.config.DBEntrySize+i]
		//	}
	}

	return ret, nil
}

// PianoPIRClient is the stateful client for PianoPIR
type PianoPIRClient struct {
	config *PianoPIRConfig

	// the master keys for the client
	//rng       *rand.Rand
	masterKey PrfKey
	longKey   []uint32

	MaxQueryNum      uint32
	FinishedQueryNum uint32

	// an upper bound of the number of queries in each chunk
	maxQueryPerChunk uint32
	QueryHistogram   []uint32

	// primary hint table
	primaryHintNum      uint32
	primaryShortTag     []uint32 // the prf short tag
	primaryParity       []uint64 // notice that we group DBEntrySize uint64 into one entry
	primaryProgramPoint []uint32 // the point that the set is programmed

	replacementIdx [][]uint32 // the replacement indices. We have one array for each chunk
	replacementVal [][]uint64 // the replacement values. We have one array for each chunk

	// backup hint table
	backupShortTag [][]uint32 // the prf short tag
	backupParity   [][]uint64 // notice that we group DBEntrySize uint64 into one entry

	// local cache
	localCache map[uint32][]uint64
}

func primaryNumParam(Q float64, ChunkSize float64, target uint32) uint32 {
	k := math.Ceil(math.Log(2) * (float64(target)))
	return uint32(k) * uint32(ChunkSize)
}

// NewPianoPIRClient is an initialization function for the client
func NewPianoPIRClient(config *PianoPIRConfig) *PianoPIRClient {

	seed := time.Now().UnixNano()
	rng := rand.New(rand.NewSource(seed))
	masterKey := RandKey(rng)
	longKey := GetLongKey((*PrfKey128)(&masterKey))
	//seed := int64(1678852332934430000)

	maxQueryNum := uint32(math.Sqrt(float64(config.DBSize)) * math.Log(float64(config.DBSize)))
	primaryHintNum := primaryNumParam(float64(maxQueryNum), float64(config.ChunkSize), config.FailureProbLog2+1) // fail prob 2^(-41)
	primaryHintNum = (primaryHintNum + config.ThreadNum - 1) / config.ThreadNum * config.ThreadNum
	maxQueryPerChunk := 3 * uint32(float64(maxQueryNum)/float64(config.SetSize))
	maxQueryPerChunk = (maxQueryPerChunk + config.ThreadNum - 1) / config.ThreadNum * config.ThreadNum

	//fmt.Printf("maxQueryNum = %v\n", maxQueryNum)
	//fmt.Printf("primaryHintNum = %v\n", primaryHintNum)
	//fmt.Printf("maxQueryPerChunk = %v\n", maxQueryPerChunk)

	masterKey = RandKey(rng)
	return &PianoPIRClient{
		config: config,

		//rng:       rng,
		masterKey: masterKey,
		longKey:   longKey,

		MaxQueryNum:      maxQueryNum,
		FinishedQueryNum: 0,
		QueryHistogram:   make([]uint32, config.SetSize),

		primaryHintNum:      primaryHintNum,
		primaryShortTag:     make([]uint32, primaryHintNum),
		primaryParity:       make([]uint64, primaryHintNum*config.DBEntrySize),
		primaryProgramPoint: make([]uint32, primaryHintNum),

		maxQueryPerChunk: maxQueryPerChunk,
		replacementIdx:   make([][]uint32, config.SetSize),
		replacementVal:   make([][]uint64, config.SetSize),

		backupShortTag: make([][]uint32, config.SetSize),
		backupParity:   make([][]uint64, config.SetSize),

		localCache: make(map[uint32][]uint64),
	}
}

// return the local storage in bytes
func (c *PianoPIRClient) LocalStorageSize() float64 {
	localStorageSize := float64(0)
	localStorageSize = localStorageSize + float64(c.primaryHintNum)*4                                // the primary hint short tag
	localStorageSize = localStorageSize + float64(c.primaryHintNum)*float64(c.config.DBEntryByteNum) // the primary parity
	localStorageSize = localStorageSize + float64(c.primaryHintNum)*4                                // the primary program point
	totalBackupHintNum := float64(c.config.SetSize) * float64(c.maxQueryPerChunk)
	localStorageSize = localStorageSize + float64(totalBackupHintNum)*4                                // the replacement indices
	localStorageSize = localStorageSize + float64(totalBackupHintNum)*float64(c.config.DBEntryByteNum) // the replacement values
	localStorageSize = localStorageSize + float64(totalBackupHintNum)*4                                // the backup short tag
	localStorageSize = localStorageSize + float64(totalBackupHintNum)*float64(c.config.DBEntryByteNum) // the backup parities

	return localStorageSize
}

func (c *PianoPIRClient) PrintStorageBreakdown() {
	fmt.Printf("primary hint short tag = %v\n", c.primaryHintNum*4)
	fmt.Printf("primary parity = %v\n", c.primaryHintNum*c.config.DBEntryByteNum)
	fmt.Printf("primary program point = %v\n", c.primaryHintNum*4)
	totalBackupHintNum := c.config.SetSize * c.maxQueryPerChunk
	fmt.Printf("replacement indices = %v\n", totalBackupHintNum*4)
	fmt.Printf("replacement values = %v\n", totalBackupHintNum*c.config.DBEntryByteNum)
	fmt.Printf("backup short tag = %v\n", totalBackupHintNum*4)
	fmt.Printf("backup parities = %v\n", totalBackupHintNum*c.config.DBEntryByteNum)
}

func (c *PianoPIRClient) Initialization() {
	//TODO: implemente the preprocessing logic
	c.FinishedQueryNum = 0

	// resample the key
	seed := time.Now().UnixNano()
	rng := rand.New(rand.NewSource(seed))
	c.masterKey = RandKey(rng)
	c.longKey = GetLongKey((*PrfKey128)(&c.masterKey))

	c.QueryHistogram = make([]uint32, c.config.SetSize)
	for i := 0; i < int(c.config.SetSize); i++ {
		c.QueryHistogram[i] = 0
	}

	// first initialize everything to be zero

	shortTagCount := uint32(0)

	c.primaryShortTag = make([]uint32, c.primaryHintNum)
	c.primaryParity = make([]uint64, c.primaryHintNum*c.config.DBEntrySize)
	c.primaryProgramPoint = make([]uint32, c.primaryHintNum)

	for i := 0; i < int(c.primaryHintNum); i++ {
		c.primaryShortTag[i] = shortTagCount
		c.primaryParity[i] = 0
		c.primaryProgramPoint[i] = DefaultProgramPoint
		shortTagCount += 1
	}

	c.replacementIdx = make([][]uint32, c.config.SetSize)
	c.replacementVal = make([][]uint64, c.config.SetSize)
	c.backupShortTag = make([][]uint32, c.config.SetSize)
	c.backupParity = make([][]uint64, c.config.SetSize)

	for i := 0; i < int(c.config.SetSize); i++ {
		c.replacementIdx[i] = make([]uint32, c.maxQueryPerChunk)
		c.replacementVal[i] = make([]uint64, c.maxQueryPerChunk*c.config.DBEntrySize)
		c.backupShortTag[i] = make([]uint32, c.maxQueryPerChunk)
		c.backupParity[i] = make([]uint64, c.maxQueryPerChunk*c.config.DBEntrySize)

		for j := 0; j < int(c.maxQueryPerChunk); j++ {
			c.replacementIdx[i][j] = DefaultProgramPoint
			c.replacementVal[i][j] = 0
			c.backupShortTag[i][j] = shortTagCount
			c.backupParity[i][j] = 0
			shortTagCount += 1
		}
	}

	// clean the cache
	c.localCache = make(map[uint32][]uint64)
}

// entrySize has to be a multiple of 4 !!!!!!!!!!!!!
func EntryXor(a []uint64, b []uint64, entrySize uint32) {

	xorSlices(a, b, int(entrySize))

	//for i := uint32(0); i < entrySize; i++ {
	//	a[i] ^= b[i]
	//	}
}

func (c *PianoPIRClient) Preprocessing(rawDB []uint64) {
	c.Initialization() // first clean everything

	//log.Printf("len(rawDB) %v\n", len(rawDB))
	//if len(rawDB) < int(c.config.ChunkSize*c.config.SetSize*c.config.DBEntrySize) {
	//append with zeros
	//rawDB = append(rawDB, make([]uint64, int(c.config.ChunkSize*c.config.SetSize*c.config.DBEntrySize)-len(rawDB))...)
	//}

	//TODO: using multiple threads
	for i := uint32(0); i < c.config.SetSize; i++ {
		start := i * c.config.ChunkSize
		//end := min((i+1)*c.config.ChunkSize, c.config.DBSize)
		end := (i + 1) * c.config.ChunkSize
		if end*c.config.DBEntrySize > uint32(len(rawDB)) {
			// in this case, we
			tmpChunk := make([]uint64, c.config.ChunkSize*c.config.DBEntrySize)
			for j := start * c.config.DBEntrySize; j < end*c.config.DBEntrySize; j++ {
				if j >= uint32(len(rawDB)) {
					tmpChunk[j-start*c.config.DBEntrySize] = 0
				} else {
					tmpChunk[j-start*c.config.DBEntrySize] = rawDB[j]
				}
			}
			c.UpdatePreprocessing(i, tmpChunk)
		} else {
			//fmt.Println("preprocessing chunk ", i, "start ", start, "end ", end)
			c.UpdatePreprocessing(i, rawDB[start*c.config.DBEntrySize:end*c.config.DBEntrySize])
		}
	}
}

func (c *PianoPIRClient) UpdatePreprocessing(chunkId uint32, chunk []uint64) {

	seed := time.Now().UnixNano()
	rng := rand.New(rand.NewSource(seed))

	if len(chunk) < int(c.config.ChunkSize*c.config.DBEntrySize) {
		fmt.Println("not enough chunk size")
		//chunk = append(chunk, make([]uint64, int(c.config.ChunkSize*c.config.DBEntrySize)-len(chunk))...)
	}

	//fmt.Printf("primary hint num = %v\n", c.primaryHintNum)

	// first enumerate all primar hints
	for i := uint32(0); i < c.primaryHintNum; i++ {
		//fmt.Println("i = ", i)
		offset := PRFEvalWithLongKeyAndTag(c.longKey, c.primaryShortTag[i], uint64(chunkId)) & (c.config.ChunkSize - 1)
		//fmt.Printf("i = %v, offset = %v\n", i, offset)
		if (i+1)*c.config.DBEntrySize > uint32(len(c.primaryParity)) {
			//fmt.Errorf("i = %v, i*c.config.DBEntrySize = %v, len(c.primaryParity) = %v", i, i*c.config.DBEntrySize, len(c.primaryParity)
			log.Fatalf("i = %v, i*c.config.DBEntrySize = %v, len(c.primaryParity) = %v", i, i*c.config.DBEntrySize, len(c.primaryParity))
		}
		EntryXor(c.primaryParity[i*c.config.DBEntrySize:(i+1)*c.config.DBEntrySize], chunk[offset*c.config.DBEntrySize:(offset+1)*c.config.DBEntrySize], c.config.DBEntrySize)
	}

	//fmt.Println("finished primary hints")

	// second enumerate all backup hints
	for i := uint32(0); i < c.config.SetSize; i++ {
		// ignore if i == chunkId
		if i == chunkId {
			continue
		}
		for j := uint32(0); j < c.maxQueryPerChunk; j++ {
			offset := PRFEvalWithLongKeyAndTag(c.longKey, c.backupShortTag[i][j], uint64(chunkId)) & (c.config.ChunkSize - 1)
			EntryXor(c.backupParity[i][j*c.config.DBEntrySize:(j+1)*c.config.DBEntrySize], chunk[offset*c.config.DBEntrySize:(offset+1)*c.config.DBEntrySize], c.config.DBEntrySize)
		}
	}

	//fmt.Println("finished backup hints")

	// finally store the replacement

	for j := uint32(0); j < c.maxQueryPerChunk; j++ {
		offset := rng.Uint32() & (c.config.ChunkSize - 1)
		c.replacementIdx[chunkId][j] = offset + chunkId*c.config.ChunkSize
		copy(c.replacementVal[chunkId][j*c.config.DBEntrySize:(j+1)*c.config.DBEntrySize], chunk[offset*c.config.DBEntrySize:(offset+1)*c.config.DBEntrySize])
	}

	//fmt.Println("finished replacement")
}

func (c *PianoPIRClient) Query(idx uint32, server *PianoPIRServer, realQuery bool) ([]uint64, error) {

	ret := make([]uint64, c.config.DBEntrySize)
	// initialize ret to be all zeros
	for i := uint32(0); i < c.config.DBEntrySize; i++ {
		ret[i] = 0
	}

	// if it's a dummy query, then just generate c.config.SetSize random numbers between 0...c.config.ChunkSize
	if !realQuery {
		set := make([]uint32, c.config.SetSize)
		for i := uint32(0); i < c.config.SetSize; i++ {
			set[i] = i*c.config.ChunkSize + (rand.Uint32() & (c.config.ChunkSize - 1))
		}
		_, err := server.PrivateQuery(set)

		return ret, err
	}

	if idx >= c.config.DBSize {
		log.Fatalf("idx %v is out of range", idx)

		// return an empty entry and an error
		return ret, fmt.Errorf("idx %v is out of range", idx)
	}

	// if the idx is in the local cache, then return the result from the local cache
	if v, ok := c.localCache[idx]; ok {
		return v, nil
	}

	// now we need to make a real query
	if c.FinishedQueryNum >= c.MaxQueryNum {
		log.Printf("fnished query = %v", c.FinishedQueryNum)
		log.Printf("max query num = %v", c.MaxQueryNum)
		log.Printf("exceed the maximum number of queries")
		return ret, fmt.Errorf("exceed the maximum number of queries")
	}

	chunkId := idx / c.config.ChunkSize
	offset := idx % c.config.ChunkSize

	if c.QueryHistogram[chunkId] >= c.maxQueryPerChunk {
		log.Printf("Too many queries in chunk %v", chunkId)
		log.Printf("Max query per chunk = %v", c.maxQueryPerChunk)
		return ret, fmt.Errorf("too many queries in chunk %v", chunkId)
	}

	// now we find the hit hint in the primary hint table

	hitId := uint32(DefaultProgramPoint)
	for i := uint32(0); i < c.primaryHintNum; i++ {
		hintOffset := PRFEvalWithLongKeyAndTag(c.longKey, c.primaryShortTag[i], uint64(chunkId)) & (c.config.ChunkSize - 1)
		if hintOffset == offset {
			// if this chunk has been programmed in this chunk before, then it shouldn't count
			if c.primaryProgramPoint[i] == DefaultProgramPoint || (c.primaryProgramPoint[i]/c.config.ChunkSize != chunkId) {
				hitId = i
				break
			}
		}
	}

	if hitId == DefaultProgramPoint {
		//log.Printf("No hit hint in the primary hint table, current idx = %v", idx)
		return ret, fmt.Errorf("no hit hint in the primary hint table")
	}

	// now we expand this hit hint to a full set
	querySet := make([]uint32, c.config.SetSize)

	for i := uint32(0); i < c.config.SetSize; i++ {
		hintOffset := PRFEvalWithLongKeyAndTag(c.longKey, c.primaryShortTag[hitId], uint64(i)) & (c.config.ChunkSize - 1)
		querySet[i] = i*c.config.ChunkSize + hintOffset
	}

	// if it's programmed, we need to enforce it
	if c.primaryProgramPoint[hitId] != DefaultProgramPoint {
		//log.Printf("hitId = %v, c.primaryProgramPoint[hitId] = %v", hitId, c.primaryProgramPoint[hitId])
		querySet[c.primaryProgramPoint[hitId]/c.config.ChunkSize] = c.primaryProgramPoint[hitId]
	}

	// now we find the first unconsumed replacement idx and val in the chunkId-th group
	inGroupIdx := uint32(c.QueryHistogram[chunkId])
	replIdx := c.replacementIdx[chunkId][inGroupIdx]
	replVal := c.replacementVal[chunkId][inGroupIdx*c.config.DBEntrySize : (inGroupIdx+1)*c.config.DBEntrySize]
	querySet[chunkId] = replIdx

	// now we make a private query
	response, err := server.PrivateQuery(querySet)

	// we revert the influence of the replacement
	EntryXor(response, replVal, c.config.DBEntrySize)
	// we also xor the original parity
	EntryXor(response, c.primaryParity[hitId*c.config.DBEntrySize:(hitId+1)*c.config.DBEntrySize], c.config.DBEntrySize)
	// now response is the answer.

	// for now just do non private query
	//response, err := server.NonePrivateQuery(idx)

	// now we need to refresh
	c.primaryShortTag[hitId] = c.backupShortTag[chunkId][inGroupIdx]
	copy(c.primaryParity[hitId*c.config.DBEntrySize:(hitId+1)*c.config.DBEntrySize], c.backupParity[chunkId][inGroupIdx*c.config.DBEntrySize:(inGroupIdx+1)*c.config.DBEntrySize])
	c.primaryProgramPoint[hitId] = idx                                                                                   // program the original index
	EntryXor(c.primaryParity[hitId*c.config.DBEntrySize:(hitId+1)*c.config.DBEntrySize], response, c.config.DBEntrySize) // also need to add the current response to the parity

	//finally we need to update the history information
	c.FinishedQueryNum += 1
	c.QueryHistogram[chunkId] += 1
	c.localCache[idx] = response

	return response, err
}

type PianoPIR struct {
	config *PianoPIRConfig
	client *PianoPIRClient
	server *PianoPIRServer
}

func NewPianoPIR(DBSize uint32, DBEntryByteNum uint32, rawDB []uint64, FailureProbLog2 uint32) *PianoPIR {
	DBEntrySize := DBEntryByteNum / 8

	// assert that the rawDB is of the correct size
	if uint32(len(rawDB)) != DBSize*DBEntrySize {
		log.Fatalf("Piano PIR len(rawDB) = %v; want %v", len(rawDB), DBSize*DBEntrySize)
	}

	targetChunkSize := uint32(2 * math.Sqrt(float64(DBSize)))
	ChunkSize := uint32(1)
	for ChunkSize < targetChunkSize {
		ChunkSize *= 2
	}
	SetSize := uint32(math.Ceil(float64(DBSize) / float64(ChunkSize)))
	// round up to the next mulitple of 4
	SetSize = (SetSize + 3) / 4 * 4

	config := &PianoPIRConfig{
		DBEntryByteNum:  DBEntryByteNum,
		DBEntrySize:     DBEntrySize,
		DBSize:          DBSize,
		ChunkSize:       ChunkSize,
		SetSize:         SetSize,
		ThreadNum:       8,
		FailureProbLog2: FailureProbLog2,
	}

	client := NewPianoPIRClient(config)
	server := NewPianoPIRServer(config, rawDB)

	return &PianoPIR{
		config: config,
		client: client,
		server: server,
	}
}

func (p *PianoPIR) Preprocessing() {
	p.client.Preprocessing(p.server.rawDB)
}

func (p *PianoPIR) Query(idx uint32, realQuery bool) ([]uint64, error) {

	if p.client.FinishedQueryNum == p.client.MaxQueryNum {
		fmt.Printf("exceed the maximum number of queries %v and redo preprocessing\n", p.client.MaxQueryNum)
		p.client.Preprocessing(p.server.rawDB)
	}

	return p.client.Query(idx, p.server, realQuery)
}

func (p *PianoPIR) LocalStorageSize() float64 {
	return p.client.LocalStorageSize()
}

func (p *PianoPIR) CommCostPerQuery() float64 {

	// upload contains p.config.SetSize 32-bit integers
	// download contains p.config.DBEntrySize 64-bit integers
	return float64(p.config.SetSize*4 + p.config.DBEntrySize*8)
}

func (p *PianoPIR) Config() *PianoPIRConfig {
	return p.config
}
