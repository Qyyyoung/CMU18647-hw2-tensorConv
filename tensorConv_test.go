package tensorConv

import (
	"fmt"
	"github.com/stretchr/testify/assert"
	"math"
	"math/rand"
	"testing"
)

func TestTensorConvWithIntSlice(t *testing.T) {
	totalSize := 256
	input := make([]int, totalSize)
	output := make([]int, totalSize)
	for i := range input {
		input[i] = rand.Int()
	}

	rank := 4
	root := float64(1) / float64(rank)
	rowSize := int (math.Pow(float64(totalSize) , root))
	for i := range input {
		MoveElementOnIntSlice(input, output, i, rank, rowSize)
	}

	rawInputIndex := []int{0, 1, 0, 1}
	rawOutputIndex := []int{1, 0, 1, 0}
	inputIndex := indexToAddress(rawInputIndex, rowSize, rank)
	outputIndex := indexToAddress(rawOutputIndex, rowSize, rank)
	assert.Equal(t, input[inputIndex], output[outputIndex])
	fmt.Printf("value at input%v is %v \n",rawInputIndex, input[inputIndex])
	fmt.Printf("value at output%v is %v \n",rawOutputIndex, output[outputIndex])

	rawInputIndex2 := []int{1, 1, 0, 0}
	rawOutputIndex2 := []int{0, 0, 1, 1}
	inputIndex2 := indexToAddress(rawInputIndex2, rowSize, rank)
	outputIndex2 := indexToAddress(rawOutputIndex2, rowSize, rank)
	assert.Equal(t, input[inputIndex2], output[outputIndex2])
	fmt.Printf("value at input%v is %v \n",rawInputIndex2, input[inputIndex2])
	fmt.Printf("value at output%v is %v \n",rawOutputIndex2, output[outputIndex2])
}


func TestTensorConvWithByteSlice(t *testing.T) {
	totalSize := 256
	input := make([]byte, totalSize)
	output := make([]byte, totalSize)
	rand.Read(input)

	rank := 4
	root := float64(1) / float64(rank)
	rowSize := int (math.Pow(float64(totalSize) , root))
	for i := range input {
		MoveElementOnByteSlice(input, output, i, rank, rowSize)
	}

	rawInputIndex := []int{0, 1, 0, 1}
	rawOutputIndex := []int{1, 0, 1, 0}
	inputIndex := indexToAddress(rawInputIndex, rowSize, rank)
	outputIndex := indexToAddress(rawOutputIndex, rowSize, rank)
	assert.Equal(t, input[inputIndex], output[outputIndex])
	fmt.Printf("value at input%v is %v \n",rawInputIndex, input[inputIndex])
	fmt.Printf("value at output%v is %v \n",rawOutputIndex, output[outputIndex])

	rawInputIndex2 := []int{1, 1, 0, 0}
	rawOutputIndex2 := []int{0, 0, 1, 1}
	inputIndex2 := indexToAddress(rawInputIndex2, rowSize, rank)
	outputIndex2 := indexToAddress(rawOutputIndex2, rowSize, rank)
	assert.Equal(t, input[inputIndex2], output[outputIndex2])
	fmt.Printf("value at input%v is %v \n",rawInputIndex2, input[inputIndex2])
	fmt.Printf("value at output%v is %v \n",rawOutputIndex2, output[outputIndex2])
}

func TestTensorConvWithParallelism(t *testing.T) {
	totalSize := 256
	input := make([]byte, totalSize)
	output := make([]byte, totalSize)
	rand.Read(input)

	rank := 4
	root := float64(1) / float64(rank)
	rowSize := int (math.Pow(float64(totalSize) , root))
	for i := range input {
		go MoveElementOnByteSlice(input, output, i, rank, rowSize)
	}

	rawInputIndex := []int{0, 1, 0, 1}
	rawOutputIndex := []int{1, 0, 1, 0}
	inputIndex := indexToAddress(rawInputIndex, rowSize, rank)
	outputIndex := indexToAddress(rawOutputIndex, rowSize, rank)
	assert.Equal(t, input[inputIndex], output[outputIndex])
	fmt.Printf("value at input%v is %v \n",rawInputIndex, input[inputIndex])
	fmt.Printf("value at output%v is %v \n",rawOutputIndex, output[outputIndex])

	rawInputIndex2 := []int{1, 1, 0, 0}
	rawOutputIndex2 := []int{0, 0, 1, 1}
	inputIndex2 := indexToAddress(rawInputIndex2, rowSize, rank)
	outputIndex2 := indexToAddress(rawOutputIndex2, rowSize, rank)
	assert.Equal(t, input[inputIndex2], output[outputIndex2])
	fmt.Printf("value at input%v is %v \n",rawInputIndex2, input[inputIndex2])
	fmt.Printf("value at output%v is %v \n",rawOutputIndex2, output[outputIndex2])
}

func indexToAddress(index []int, rowSize int, rank int) int {
	result := 0
	for i, val := range index {
		result += int (math.Pow(float64(rowSize),float64(rank - i - 1)) ) * val
	}
	return result
}

