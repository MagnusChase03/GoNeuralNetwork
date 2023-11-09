// The neuralnetwork package is used
// to define the structure and behavior
// of a neural network
package neuralnetwork

import (
	"fmt"
	"os"
	"bytes"
	"encoding/gob"

	"github.com/MagnusChase03/gonn/layer"
)

// The NeuralNetwork struct is used
// to hold the information of a nerual network
type NeuralNetwork struct {
	LearningRate  float64
	Shape         []int
	Layers        []layer.Layer
}

// The CreateNeuralNetwork function instansiates
// a new neural network
func CreateNeuralNetwork(shapes [][]int, learningRate float64) (*NeuralNetwork, error) {
    neuralnetwork := new(NeuralNetwork)
    neuralnetwork.LearningRate = learningRate

    neuralnetwork.Shape = make([]int,2,2)
    neuralnetwork.Shape[0] = shapes[0][0]
    neuralnetwork.Shape[1] = shapes[len(shapes) - 1][1]

    neuralnetwork.Layers = make([]layer.Layer,len(shapes),len(shapes))
    for i := 0; i < len(shapes); i++ {
        if i > 0 && shapes[i - 1][1] != shapes[i][0] {
            return nil, fmt.Errorf("dimension error")
        }

        neuralnetwork.Layers[i] = layer.CreateDenseLayer(shapes[i][0], shapes[i][1])
    }

    return neuralnetwork, nil
}

// Implementing the Forward method for the
// NeuralNetwork
func (n *NeuralNetwork) Forward(inputs []float64) error {
	for i := 0; i < len(n.Layers); i++ {
		var values []float64
		if i == 0 {
			values = inputs
		} else {
			values = n.Layers[i - 1].GetOutputs()
		}

		if err := n.Layers[i].Forward(values); err != nil {
			return err
		}
	}

	return nil
}

// Implementing the Backward method for the
// NeuralNetwork
func (n *NeuralNetwork) Backward(outputs []float64) error {
	for i := len(n.Layers) - 1; i >= 0; i-- {
		hiddenLayer := i < len(n.Layers) - 1
		var values []float64
		if i == len(n.Layers) - 1 {
			values = outputs
		} else {
			values = n.Layers[i + 1].GetDeltaInputs()
		}

		if err := n.Layers[i].Backward(values, n.LearningRate, hiddenLayer); err != nil {
			return err
		}
	}

	return nil
}

// Implementing the Update method for the
// NeuralNetwork
func (n *NeuralNetwork) Update() {
	for i := 0; i < len(n.Layers); i++ {
		n.Layers[i].Update()
	}
} 

// Save stores the entire state of the 
// neural network into a file on disk
func (n *NeuralNetwork) Save(filepath string) error {
	gob.Register(&NeuralNetwork{})
	gob.Register(&layer.DenseLayer{})

	content := &bytes.Buffer{}
	enc := gob.NewEncoder(content)
	if err := enc.Encode(n); err != nil {
		return err
	}

	if err := os.WriteFile(filepath, content.Bytes(), 0644); err != nil {
		return err
	}

	return nil
}

// Load loads the entire state of the 
// neural network from a json file on disk
func Load(filepath string) (*NeuralNetwork, error) {
	content, err := os.ReadFile(filepath)
	if err != nil {
		return nil, err
	}

	neuralnetwork := new(NeuralNetwork)
	buffer := bytes.NewBuffer(content)

	gob.Register(&NeuralNetwork{})
	gob.Register(&layer.DenseLayer{})
	dec := gob.NewDecoder(buffer)
	if err := dec.Decode(&neuralnetwork); err != nil {
		return nil, err
	}

	return neuralnetwork, nil
}