// The layer package is used to define
// the structure of neural network layers
// and how they work
package layer

import (
	"fmt"
	"math"
	"math/rand"
)

// Layer is the interace which defines 
// the behaviors all layers must implement
type Layer interface {
	Forward(inputs []float64) error
	Backward(
		outputs []float64, 
		learningRate float64,
		hiddenLayer bool,
	) error
	Update()

	GetType()            string
	GetDeltaInputs()  []float64
	GetOutputs()      []float64
}

// DenseLayer is the data structure for
// holding information about a dense layer
type DenseLayer struct {
    Shape           []int
	Type            string

    Inputs          []float64
    Outputs         []float64

    Weights         [][]float64
    Bias            []float64
    DeltaInputs     []float64
    DeltaWeights    [][]float64
    DeltaBias       []float64
}

// CreateDenseLayer instansiates a
// dense layer with random weights
func CreateDenseLayer(inputs int, outputs int) *DenseLayer {
    denseLayer := new(DenseLayer)

	denseLayer.Type = "DenseLayer"
    denseLayer.Shape = make([]int,2,2)
    denseLayer.Shape[0] = inputs
    denseLayer.Shape[1] = outputs

    denseLayer.Inputs = make([]float64,inputs,inputs)
    denseLayer.DeltaInputs = make([]float64,inputs,inputs)

    denseLayer.Outputs = make([]float64,outputs,outputs)
    denseLayer.Bias = make([]float64,outputs,outputs)
    denseLayer.DeltaBias = make([]float64,outputs,outputs)

    denseLayer.Weights = make([][]float64,inputs,inputs)
    denseLayer.DeltaWeights = make([][]float64,inputs,inputs)
    for i := 0; i < inputs; i++ {
        denseLayer.Weights[i] = make([]float64,outputs,outputs)
        for j := 0; j < outputs; j++ {
            denseLayer.Weights[i][j] = rand.Float64() / 100
        }
        denseLayer.DeltaWeights[i] = make([]float64,outputs,outputs)
    }

    return denseLayer
}

// Sigmoid is an activation function
func sigmoid(x float64) float64 {
    return 1 / (1 + math.Exp(-x))
}

// Implementing the Forward method for
// the DenseLayer
func (d *DenseLayer) Forward(inputs []float64) error {
	if len(inputs) != d.Shape[0] {
		return fmt.Errorf("dimension error")
	}

	for i := 0; i < d.Shape[0]; i++ {
		d.Inputs[i] = inputs[i]
	}

	for i := 0; i < d.Shape[1]; i++ {
		total := 0.0
		for j := 0; j < d.Shape[0]; j++ {
			total += d.Inputs[j] * d.Weights[j][i]
		}
		total += d.Bias[i]
		d.Outputs[i] = sigmoid(total)
	}

	return nil
}

// Implementing the Backward method for
// the DenseLayer
func (d *DenseLayer) Backward(outputs []float64, learningRate float64, hiddenLayer bool) error {
	if len(outputs) != d.Shape[1] {
		return fmt.Errorf("dimension error")
	}

	for i := 0; i < d.Shape[1]; i++ {
		var delta float64
		derivative := d.Outputs[i] * (1 - d.Outputs[i])
		if hiddenLayer {
			delta = outputs[i] * derivative * learningRate
		} else {
			delta = 2 * (outputs[i] - d.Outputs[i]) * derivative * learningRate
		}

		d.DeltaBias[i] += delta
		for j := 0; j < d.Shape[0]; j++ {
			d.DeltaInputs[j] += delta * d.Weights[j][i]
			d.DeltaWeights[j][i] += delta * d.Inputs[j]
		}
	}

	return nil
}

// Implementing the Update method for
// the DenseLayer
func (d *DenseLayer) Update() {
	for i := 0; i < d.Shape[1]; i++ {
		d.Bias[i] += d.DeltaBias[i]
		d.DeltaBias[i] = 0.0
		for j := 0; j < d.Shape[0]; j++ {
			d.Weights[j][i] += d.DeltaWeights[j][i]
			d.DeltaWeights[j][i] = 0.0

			if i == 0 {
				d.DeltaInputs[j] = 0.0
			}
		}
	}
}

// Implements the Get Outputs funciton
// for the DenseLayer
func (d *DenseLayer) GetOutputs() []float64 {
	return d.Outputs
}

// Implements the Get Delta Inputs 
// function for the DenseLayer
func (d *DenseLayer) GetDeltaInputs() []float64 {
	return d.DeltaInputs
}

// Implements the GetType function
// for the DenseLayer
func (d *DenseLayer) GetType() string {
	return d.Type
}