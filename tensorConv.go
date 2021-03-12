package tensorConv

import (
	"math"
)

/*
	This code performs basic function that implements
	tensor conversion algorithm
*/

func MoveElementOnIntSlice(input []int, output []int, address int, rank int, rowSize int) {
	result := GetDestinationIndex(address, rank, rowSize)
	output[result] = input[address]
}

func MoveElementOnByteSlice(input []byte, output []byte, address int, rank int, rowSize int) {
	result := GetDestinationIndex(address, rank, rowSize)
	output[result] = input[address]
}

func GetDestinationIndex(address int, rank int, rowSize int) int {
	// index in slice is shown as [index_(rank), index_(rank-1), ..., index_1, index_0]
	// in other words, index 0 of Index slice stores the actual index for highest dimension
	cIndex := make([]int, rank)
	fortanIndex := make([]int, rank)

	// calculate the rank-dimensional representation in C index
	var remainder, level, levelFactor int
	remainder = address
	level = rank - 1
	// saves some operations here by existing when remainder is 0
	for level > 0 && remainder != 0 {
		// since the tensor has equal length in dimensions, we can calculate the factor easier
		levelFactor = int(math.Pow(float64(rowSize), float64(level)))
		cIndex[rank - level - 1] = int (math.Floor(float64(remainder / levelFactor)))
		remainder = remainder % levelFactor
		level -= 1
	}

	// fill the rest of cIndex with zero
	for level > 0 {
		cIndex[rank - level - 1] = 0
		level -= 1
	}

	// place the last index
	cIndex[rank - level - 1] = remainder

	// reverse the index into fortanIndex,
	// in other words, flip the index into fortan format
	for j := 0; j < rank; j ++ {
		fortanIndex[j] = cIndex[rank- j -1]
	}

	// calculate the linear address according to fortanIndex
	result := 0
	for i, index := range fortanIndex {
		result += int (math.Pow(float64(rowSize),float64(rank - i - 1)) ) * index
	}
	return result
}