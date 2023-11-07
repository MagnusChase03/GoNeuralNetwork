package layer

import (
    "os"
    "math"
    "math/rand"
    "errors"
    "encoding/json"
)

type Layer struct {
    Shape           []int

    Inputs          []float64
    Outputs         []float64

    Weights         [][]float64
    Bias            []float64
    Delta_Inputs    []float64
    Delta_Weights   [][]float64
    Delta_Bias      []float64
}

func Create_Layer(inputs int, outputs int) *Layer {
    layer := new(Layer)

    layer.Shape = make([]int,2,2)
    layer.Shape[0] = inputs
    layer.Shape[1] = outputs

    layer.Inputs = make([]float64,inputs,inputs)
    layer.Delta_Inputs = make([]float64,inputs,inputs)

    layer.Outputs = make([]float64,outputs,outputs)
    layer.Bias = make([]float64,outputs,outputs)
    layer.Delta_Bias = make([]float64,outputs,outputs)

    layer.Weights = make([][]float64,inputs,inputs)
    layer.Delta_Weights = make([][]float64,inputs,inputs)
    for i := 0; i < inputs; i++ {
        layer.Weights[i] = make([]float64,outputs,outputs)
        for j := 0; j < outputs; j++ {
            layer.Weights[i][j] = rand.Float64() / 100
        }
        layer.Delta_Weights[i] = make([]float64,outputs,outputs)
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
        layer.Outputs[i] = Sigmoid(total + layer.Bias[i])
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
        for j := 0; j < layer.Shape[1]; j++ {
            if (hidden_layer) {
                layer.Delta_Inputs[i] += outputs[j] * 
                                         layer.Outputs[j] * (1 - layer.Outputs[j]) * 
                                         layer.Weights[i][j] * learning_rate

                layer.Delta_Weights[i][j] += outputs[j] * 
                                             layer.Outputs[j] * (1 - layer.Outputs[j]) * 
                                             layer.Inputs[i] * learning_rate

                if (i == 0) {
                    layer.Delta_Bias[j] += outputs[j] * 
                                           layer.Outputs[j] * (1 - layer.Outputs[j]) * 
                                           learning_rate
                }
            } else {
                layer.Delta_Inputs[i] += 2 * (outputs[j] - layer.Outputs[j]) * 
                                         layer.Outputs[j] * (1 - layer.Outputs[j]) * 
                                         layer.Weights[i][j] * learning_rate

                layer.Delta_Weights[i][j] += 2 * (outputs[j] - layer.Outputs[j]) * 
                                             layer.Outputs[j] * (1 - layer.Outputs[j]) * 
                                             layer.Inputs[i] * learning_rate

                if (i == 0) {
                    layer.Delta_Bias[j] += 2 * (outputs[j] - layer.Outputs[j]) * 
                                           layer.Outputs[j] * (1 - layer.Outputs[j]) * 
                                           learning_rate
                }
            }
        }
    }

    return nil
}

func Update(layer *Layer) {
    for i := 0; i < layer.Shape[0]; i++ {
        layer.Delta_Inputs[i] = 0.0
        for j := 0; j < layer.Shape[1]; j++ {
            layer.Weights[i][j] += layer.Delta_Weights[i][j]
            layer.Delta_Weights[i][j] = 0.0

            if (i == 0) {
                layer.Bias[j] += layer.Delta_Bias[j]
                layer.Delta_Bias[j] = 0.0
            }
        }
    } 
}

func Save(layer *Layer, filepath string) error {
    f, err := os.Create(filepath)
    if (err != nil) {
        return err
    }

    defer f.Close()

    data, err := json.Marshal(*layer)
    if (err != nil) {
        return err
    }

    _, err = f.Write(data)
    if (err != nil) {
        return err
    }

    return nil
}
