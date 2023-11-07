package layer

import (
    "math"
    "math/rand"
    "errors"
)

type Layer struct {
    Shape    []int

    Inputs   []float64
    Cache    []float64
    Outputs  []float64

    Weights  [][]float64
    Bias     []float64
    Delta    []float64
}

func Create_Layer(inputs int, outputs int) *Layer {
    layer := new(Layer)

    layer.Shape = make([]int,2,2)
    layer.Shape[0] = inputs
    layer.Shape[1] = outputs

    layer.Inputs = make([]float64,inputs,inputs)
    layer.Delta = make([]float64,inputs,inputs)

    layer.Cache = make([]float64,outputs,outputs)
    layer.Outputs = make([]float64,outputs,outputs)
    layer.Bias = make([]float64,outputs,outputs)

    layer.Weights = make([][]float64,inputs,inputs)
    for i := 0; i < inputs; i++ {
        layer.Weights[i] = make([]float64,outputs,outputs)
        for j := 0; j < outputs; j++ {
            layer.Weights[i][j] = rand.Float64() / 100
        }
    }

    return layer
}

func Sigmoid(x float64) float64 {
    return 1 / (1 + math.Exp(-x))
}

func Forward(layer *Layer, inputs []float64) error {
    if (len(inputs) != layer.Shape[0]) {
        return errors.New("== Dimension Error ==")
    }

    for i := 0; i < layer.Shape[0]; i++ {
        layer.Inputs[i] = inputs[i]
    }

    for i := 0; i < layer.Shape[1]; i++ {
        var total float64
        for j := 0; j < layer.Shape[0]; j++ {
            total += layer.Weights[j][i] * inputs[j]
        }
        layer.Cache[i] = total + layer.Bias[i]
        layer.Outputs[i] = Sigmoid(layer.Cache[i])
    }

    return nil 
}

func Get_Error(layer *Layer, outputs []float64) float64 {
    var error float64;
    for i := 0; i < layer.Shape[1]; i++ {
        error += math.Pow((outputs[i] - layer.Outputs[i]), 2)
    }

    return error
}

func Backward(layer *Layer, outputs []float64, learning_rate float64, hidden_layer bool) error {
    if (len(outputs) != layer.Shape[1]) {
        return errors.New("== Dimension Error ==")
    }

    for i := 0; i < layer.Shape[0]; i++ {
        layer.Delta[i] = 0.0
        for j := 0; j < layer.Shape[1]; j++ {
            if (hidden_layer) {
                layer.Delta[i] += outputs[j] * 
                                  layer.Outputs[j] * (1 - layer.Outputs[j]) * 
                                  layer.Weights[i][j] * learning_rate

                layer.Weights[i][j] += outputs[j] * 
                                       layer.Outputs[j] * (1 - layer.Outputs[j]) * 
                                       layer.Inputs[i] * learning_rate
            } else {
                layer.Delta[i] += 2 * (outputs[j] - layer.Outputs[j]) * 
                                  layer.Outputs[j] * (1 - layer.Outputs[j]) * 
                                  layer.Weights[i][j] * learning_rate

                layer.Weights[i][j] += 2 * (outputs[j] - layer.Outputs[j]) * 
                                       layer.Outputs[j] * (1 - layer.Outputs[j]) * 
                                       layer.Inputs[i] * learning_rate
            }
        }
    }

    return nil
}
