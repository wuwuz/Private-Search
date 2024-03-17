package pianopir

import (
	"math/rand"
	"testing"
	"time"
)

func TestPIRBasic(t *testing.T) {
	// Arrange
	// Set up any necessary data or arguments

	DBSize := uint32(187500)
	DBEntrySize := uint32(4)
	seed := time.Now().UnixNano()
	rng := rand.New(rand.NewSource(seed))

	rawDB := make([]uint64, DBEntrySize*DBSize)
	for i := uint32(0); i < DBSize; i++ {
		for j := uint32(0); j < DBEntrySize; j++ {
			rawDB[i*DBEntrySize+j] = rng.Uint64()
		}
	}

	PIR := NewPianoPIR(DBSize, DBEntrySize*8, rawDB, 40)

	// print the config of the PIR
	config := PIR.Config()
	t.Logf("PIR config: %v", config)
	t.Logf("hint num: %v", PIR.client.primaryHintNum)
	t.Logf("max query num: %v", PIR.client.MaxQueryNum)

	//_maxQueryNum := PIR.client.MaxQueryNum

	PIR.Preprocessing()
	t.Logf("finished preprocessing")
	t.Logf("PIR storage %v MB", PIR.LocalStorageSize()/1024/1024)

	PIR.verifyPreprocessing()
	t.Logf("finished verifying preprocessing")

	// make 1000 random queries
	//for i := 0; i < int(maxQueryNum); i++ {
	errorFlag := false
	for i := 0; i < int(1000); i++ {
		idx := rand.Uint32() % DBSize
		query, err := PIR.Query(idx, true)
		if err != nil {
			t.Errorf("PIR.Query(%v) failed: %v", idx, err)
		}

		for j := uint32(0); j < DBEntrySize; j++ {
			if query[j] != rawDB[idx*DBEntrySize+j] {
				t.Errorf("query[%v] = %v; want %v", idx, query[j], rawDB[idx*DBEntrySize+j])
				errorFlag = true
			}
		}
		if errorFlag {
			t.Logf("response = %v", query)
			break
		}

		if i%100 == 0 {
			t.Logf("The (%v)-th PIR.Query(%v) passed", i, idx)
		}

		// just output a message to show the progress
		//t.Logf("PIR.Query(%v) passed", idx)
	}
}

func TestBatchPIRBasic(t *testing.T) {
	// Arrange
	// Set up any necessary data or arguments

	DBSize := uint32(1000000)
	DBEntrySize := uint32(16)
	BatchSize := uint32(32)

	// a seed that's depending on the current time
	//seed := time.Now().UnixNano()
	//rng := rand.New(rand.NewSource(seed))

	rawDB := make([]uint64, DBEntrySize*DBSize)
	for i := uint32(0); i < DBSize; i++ {
		for j := uint32(0); j < DBEntrySize; j++ {
			rawDB[i*DBEntrySize+j] = uint64(i) //rng.Uint64()
		}
	}

	PIR := NewSimpleBatchPianoPIR(DBSize, DBEntrySize*8, BatchSize, rawDB, 20)

	// print the config of the PIR
	config := PIR.Config()
	t.Logf("Batch PIR config: %v", config)

	PIR.Preprocessing()

	// make a single batch query
	// for each partition, make PartitionQueryNum queries
	batchQuery := make([]uint32, 0, BatchSize)

	for i := uint32(0); i < config.PartitionNum; i++ {
		start := i * config.PartitionSize
		end := min((i+1)*config.PartitionSize, DBSize)

		for j := uint32(0); j < QueryPerPartition-1; j++ {
			offset := rand.Uint32() % (end - start)
			// append the query to the batch query
			batchQuery = append(batchQuery, start+offset)
		}
	}

	// now make a batch query
	// they should be all correct

	responses, err := PIR.Query(batchQuery)

	if err != nil {
		t.Errorf("PIR.Query(%v) failed: %v", batchQuery, err)
	}

	for i := 0; i < len(batchQuery); i++ {
		idx := batchQuery[i]
		query := responses[i]
		for j := uint32(0); j < DBEntrySize; j++ {
			if query[j] != rawDB[idx*DBEntrySize+j] {
				t.Errorf("query[%v] = %v; want %v", idx, query[j], rawDB[idx*DBEntrySize+j])
			}
		}
	}

	//t.Logf("Batch PIR.Query(%v) passed", batchQuery)

	// now make another batch query
	// it only has queries in the first partition
	// so only the first PartitionQueryNum queries should be correct

	querySet := make(map[uint32]bool)
	batchQuery = make([]uint32, 0, BatchSize)
	for i := uint32(0); i < BatchSize; i++ {
		idx := rand.Uint32() % config.PartitionSize
		if _, ok := querySet[idx]; ok {
			// resample the index
			i--
			continue
		}
		querySet[idx] = true
		batchQuery = append(batchQuery, idx)
	}

	// now make a batch query
	// only the first PartitionQueryNum queries should be correct
	// the rest should be all zeros

	//fmt.Println("batchQuery: ", batchQuery)

	responses, err = PIR.Query(batchQuery)

	if err != nil {
		t.Errorf("PIR.Query(%v) failed: %v", batchQuery, err)
	}

	for i := uint32(0); i < BatchSize; i++ {
		idx := batchQuery[i]
		query := responses[i]

		if i < QueryPerPartition {
			// check if the first PartitionQueryNum queries are correct
			for j := uint32(0); j < DBEntrySize; j++ {
				if query[j] != rawDB[idx*DBEntrySize+j] {
					t.Errorf("query[%v] = %v; want %v", idx, query[j], rawDB[idx*DBEntrySize+j])
				}
			}
		} else {
			// otherwise check if they are all zeros
			for j := uint32(0); j < DBEntrySize; j++ {
				if query[j] != 0 {
					t.Errorf("query[%v] = %v; want 0", idx, query[j])
				}
			}
		}
	}
}

func TestBatchPIRPerf(t *testing.T) {
	// Arrange
	// Set up any necessary data or arguments

	DBSize := uint32(3201821)
	//DBSize := uint32(300000)
	DBEntrySize := uint32(112)
	BatchSize := uint32(32)

	// a seed that's depending on the current time
	seed := time.Now().UnixNano()
	rng := rand.New(rand.NewSource(seed))

	rawDB := make([]uint64, DBEntrySize*DBSize)
	for i := uint32(0); i < DBSize; i++ {
		for j := uint32(0); j < DBEntrySize; j++ {
			rawDB[i*DBEntrySize+j] = rng.Uint64()
		}
	}

	PIR := NewSimpleBatchPianoPIR(DBSize, DBEntrySize*8, BatchSize, rawDB, 8)

	// print the config of the PIR
	config := PIR.Config()
	t.Logf("Batch PIR config: %v\n", config)
	t.Logf("Batch PIR storage %v MB\n", PIR.LocalStorageSize()/1024/1024)
	t.Logf("Batch PIR max query num%v\n", PIR.subPIR[0].client.MaxQueryNum)
	t.Logf("Sub PIR config: %v\n", PIR.subPIR[0].Config())
	t.Logf("Sub PIR primary hint num: %v\n", PIR.subPIR[0].client.primaryHintNum)
	t.Logf("Sub PIR strorae %v MB\n", PIR.subPIR[0].LocalStorageSize()/1024/1024)
	PIR.subPIR[0].client.PrintStorageBreakdown()

	start := time.Now()
	PIR.Preprocessing()
	end := time.Now()
	t.Logf("Preprocessing time = %v\n", end.Sub(start))

	// now we make 1000 random batchQuery

	step := 20
	queryNum := 300

	start = time.Now()
	for i := 0; i < queryNum*step; i++ {
		batch := make([]uint32, 0, BatchSize)
		for j := 0; j < int(BatchSize); j++ {
			batch = append(batch, rng.Uint32()%DBSize)
		}
		response, err := PIR.Query(batch)
		if err != nil {
			t.Errorf("PIR.Query(%v) failed: %v", batch, err)
		}
		//we check the first response, either it's all zeros, or it's correct
		for j := uint32(0); j < DBEntrySize; j++ {
			if response[0][j] != 0 && response[0][j] != rawDB[batch[0]*DBEntrySize+j] {
				t.Errorf("response[0][%v] = %v; want %v", j, response[0][j], rawDB[batch[0]*DBEntrySize+j])
			}
		}
	}
	end = time.Now()
	t.Logf("Total query time = %v\n", end.Sub(start))
	t.Logf("Average query time per batch = %v\n", end.Sub(start)/time.Duration(queryNum*step))
	t.Logf("Average query time given all steps = %v\n", end.Sub(start)/time.Duration(queryNum))
}

func TestXORPerf(t *testing.T) {

	p := make([]uint64, 8)
	q := make([]uint64, 8)
	for i := 0; i < 8; i++ {
		p[i] = 12312312
		q[i] = 12312
	}
	xorSlices(p, q, 8)
	for i := 0; i < 8; i++ {
		if p[i] != 12312312^12312 {
			t.Errorf("p[%v] = %v; want %v", i, p[i], 12312312^12312)
		}
	}

	n := 1000000
	l := 112
	a := make([]uint64, l*n)
	b := make([]uint64, l*n)

	for i := 0; i < n; i++ {
		for j := 0; j < l; j++ {
			a[i*l+j] = 12312312
			b[i*l+j] = 12312
		}
	}

	// naive xor

	start := time.Now()
	for i := 0; i < n; i++ {
		for j := 0; j < l; j++ {
			a[i*l+j] ^= b[i*l+j]
		}
	}
	end := time.Now()
	t.Logf("Naive XOR time = %v\n", end.Sub(start))

	for i := 0; i < l*n; i++ {
		a[i] = 12312312
		b[i] = 12312
	}

	// use XorSlice
	start = time.Now()
	xorSlices(a, b, l*n)
	end = time.Now()
	t.Logf("XorSlices time = %v\n", end.Sub(start))

	// verify the result
	for i := 0; i < l*n; i++ {
		if a[i] != 12312312^12312 {
			t.Errorf("a[%v] = %v; want %v", i, a[i], 12312312^12312)
		}
	}
}

func TestAESPerf(t *testing.T) {

	seed := time.Now().UnixNano()
	rng := rand.New(rand.NewSource(seed))
	masterKey := RandKey(rng)
	longKey := GetLongKey((*PrfKey128)(&masterKey))

	n := 1000000
	tag := make([]uint32, n)
	results := make([]uint32, n)

	for i := 0; i < n; i++ {
		tag[i] = rng.Uint32()
		results[i] = 0
	}

	start := time.Now()
	for i := 0; i < n; i++ {
		results[i] = PRFEvalWithLongKeyAndTag(longKey, tag[i], uint64(i))
	}
	end := time.Now()
	t.Logf("PRFEvalWithLongKeyAndTag time = %v\n", end.Sub(start))
	t.Logf("average time = %v ns", end.Sub(start).Nanoseconds()/int64(n))

	l := 112
	a := make([]uint64, l*n)
	b := make([]uint64, l*n)

	for i := 0; i < l*n; i++ {
		a[i] = 12312312
		b[i] = 12312
	}

	// use XorSlice
	start = time.Now()

	for i := 0; i < n; i++ {
		xorSlices(a[i*l:(i+1)*l], b[i*l:(i+1)*l], l)
	}

	end = time.Now()
	t.Logf("XorSlices time = %v\n", end.Sub(start))
	t.Logf("average time = %v ns", end.Sub(start).Nanoseconds()/int64(n))
}

func TestAESPacked(t *testing.T) {
	seed := time.Now().UnixNano()
	rng := rand.New(rand.NewSource(seed))
	masterKey := RandKey(rng)
	longKey := GetLongKey((*PrfKey128)(&masterKey))

	n := 1000000
	tag := make([]uint32, n)
	resultsPacked := make([]uint32, n)
	resultsExtracted := make([]uint32, n)

	for i := 0; i < n; i++ {
		tag[i] = uint32(i)
	}

	start := time.Now()
	for i := 0; i < n; i += 4 {
		ret := PRFEvalWithLongKeyAndTagPacked(longKey, tag[i], 0)
		resultsPacked[i], resultsPacked[i+1], resultsPacked[i+2], resultsPacked[i+3] = ret[0], ret[1], ret[2], ret[3]
	}
	end := time.Now()
	t.Logf("PRFEvalWithLongKeyAndTagPacked time = %v\n", end.Sub(start))
	t.Logf("average time = %v ns", end.Sub(start).Nanoseconds()/int64(n))

	start = time.Now()
	for i := 0; i < n; i++ {
		resultsExtracted[i] = PRFEvalWithLongKeyAndTagExtracted(longKey, tag[i], 0)
	}
	end = time.Now()
	t.Logf("PRFEvalWithLongKeyAndTagExtracted time = %v\n", end.Sub(start))
	t.Logf("average time = %v ns", end.Sub(start).Nanoseconds()/int64(n))

	for i := 0; i < n; i++ {
		if resultsPacked[i] != resultsExtracted[i] {
			t.Errorf("resultsPacked[%v] = %v; want %v", i, resultsPacked[i], resultsExtracted[i])
		}
	}
}
