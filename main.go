package main

import (
    "fmt"
    "github.com/MagnusChase03/GoNN/neuralnetwork"
)

func main() {
    //network := neuralnetwork.Create_NeuralNetwork([][]int{{3, 2}, {2, 2}}, 0.1)
    network := neuralnetwork.Load("./data/network.json")
    inputs := []float64{1.0, 2.0, 3.0}
    outputs := []float64{1.0, 0.0}
    neuralnetwork.Forward(network, inputs)

    for i := 0; i < 10000; i++ {
        neuralnetwork.Backward(network, outputs)
        neuralnetwork.Update(network)
        neuralnetwork.Forward(network, inputs)
    }

    err := neuralnetwork.Save(network, "./data/network.json")
    if (err != nil) {
        fmt.Println(err)
    }

    fmt.Println(network.Layers[len(network.Layers) - 1].Outputs)
}
