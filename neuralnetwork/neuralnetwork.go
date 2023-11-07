package neuralnetwork

import (
    "os"
    "errors"
    "github.com/MagnusChase03/GoNN/layer"
    "encoding/json"
)

type NeuralNetwork struct {
    Learning_Rate  float64

    Shape          []int
    Layers         []*layer.Layer
}

func Create_NeuralNetwork(shapes [][]int, leanring_rate float64) *NeuralNetwork {
    neuralnetwork := new(NeuralNetwork)
    neuralnetwork.Learning_Rate = leanring_rate

    neuralnetwork.Shape = make([]int,2,2)
    neuralnetwork.Shape[0] = shapes[0][0]
    neuralnetwork.Shape[1] = shapes[len(shapes) - 1][1]

    neuralnetwork.Layers = make([]*layer.Layer,len(shapes),len(shapes))
    for i := 0; i < len(shapes); i++ {
        if (i > 0 && shapes[i][1] != shapes[i][0]) {
            return nil
        }

        neuralnetwork.Layers[i] = layer.Create_Layer(shapes[i][0], shapes[i][1])
    }

    return neuralnetwork
}

func Forward(neuralnetwork *NeuralNetwork, inputs []float64) error {
    if (len(inputs) != neuralnetwork.Shape[0]) {
        return errors.New("== Dimension Error ==")
    }

    layer.Forward(neuralnetwork.Layers[0], inputs)
    for i := 1; i < len(neuralnetwork.Layers); i++ {
       layer.Forward(neuralnetwork.Layers[i], neuralnetwork.Layers[i - 1].Outputs) 
    }

    return nil
}

func Backward(neuralnetwork *NeuralNetwork, outputs []float64) error {
    if (len(outputs) != neuralnetwork.Shape[1]) {
        return errors.New("== Dimension Error ==")
    }

    layer.Backward(neuralnetwork.Layers[len(neuralnetwork.Layers) - 1], outputs, neuralnetwork.Learning_Rate, false)
    for i := len(neuralnetwork.Layers) - 2; i >= 0; i-- {
        layer.Backward(neuralnetwork.Layers[i], neuralnetwork.Layers[i + 1].Delta_Inputs, neuralnetwork.Learning_Rate, true)
    }

    return nil
}

func Update(neuralnetwork *NeuralNetwork) {
    for i := 0; i < len(neuralnetwork.Layers); i++ {
        layer.Update(neuralnetwork.Layers[i])
    }
}

func Save(neuralnetwork *NeuralNetwork, filepath string) error {

    f, err := os.Create(filepath)
    if (err != nil) {
        return err
    }

    defer f.Close()

    data, err := json.MarshalIndent(*neuralnetwork, "", "    ")
    if (err != nil) {
        return err
    }

    _, err = f.Write(data)
    if (err != nil) {
        return err
    }

    return nil

    return nil
}
