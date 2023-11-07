package main

import (
    "fmt"
    "github.com/MagnusChase03/GoNN/nerualnetwork"
)

func main() {
    network := nerualnetwork.Create_NerualNetwork([][]int{{3, 2}, {2, 2}}, 0.1)
    inputs := []float64{1.0, 2.0, 3.0}
    outputs := []float64{1.0, 0.0}
    nerualnetwork.Forward(network, inputs)

    for i := 0; i < 10000; i++ {
        nerualnetwork.Backward(network, outputs)
        nerualnetwork.Forward(network, inputs)
    }

    fmt.Println(network.Layers[len(network.Layers) - 1].Outputs)
}
