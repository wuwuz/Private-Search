package main

import (
	"math"
	"math/rand"
	"testing"
)

func TestConvert(t *testing.T) {

	// first generate a random float32 matrix with size 20x192
	matrix := make([][]float32, 20)
	for i := 0; i < 20; i++ {
		matrix[i] = make([]float32, 192)
		for j := 0; j < 192; j++ {
			matrix[i][j] = rand.Float32()
		}
	}

	// then generate a random uint32 matrix with size 20x32
	graph := make([][]uint32, 20)
	for i := 0; i < 20; i++ {
		graph[i] = make([]uint32, 32)
		for j := 0; j < 32; j++ {
			graph[i][j] = rand.Uint32()
		}
	}

	_, DBEntryByteNum, rawDB := MakeRawDB(matrix, graph)

	// now we check if the conversion is correct, row by row
	for i := uint64(0); i < 20; i++ {
		rowDBSlice := rawDB[i*DBEntryByteNum/8 : (i+1)*DBEntryByteNum/8]
		vector, neighbors := ConvertFromRawDB(192, 32, rowDBSlice)

		// now verify the conversion

		//for the vector, check the absolute error is within 1e-6
		for j := 0; j < 192; j++ {
			if math.Abs(float64(vector[j]-matrix[i][j])) > 1e-6 {
				t.Errorf("vector[%v][%v] = %v; want %v", i, j, vector[j], matrix[i][j])
			}
		}

		//for the neighbors, check if they are equal
		for j := 0; j < 32; j++ {
			if neighbors[j] != graph[i][j] {
				t.Errorf("neighbors[%v][%v] = %v; want %v", i, j, neighbors[j], graph[i][j])
			}
		}
	}
}
