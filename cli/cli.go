package cli

import (
    //"fmt"
    "os"
    "errors"
    "encoding/json"
    "github.com/MagnusChase03/GoNN/neuralnetwork"
)

type NeuralNetwork_Structure struct {
    Learning_Rate  float64
    Layer_Shapes   [][]int
}

func Execute(args []string) error {
    if (args[0] == "create") {
        err := Create(args[1])
        return err
    }

    return errors.New("Invalid Command.")
}

func Create(name string) error {
    neuralnetwork_structure := new(NeuralNetwork_Structure)

    content, err := os.ReadFile("GoNN.json")
    if (err != nil) {
        return err
    }

    err = json.Unmarshal(content, neuralnetwork_structure)
    if (err != nil) {
        return err
    }

    network := neuralnetwork.Create_NeuralNetwork(
                    neuralnetwork_structure.Layer_Shapes,
                    neuralnetwork_structure.Learning_Rate)

    os.Mkdir("./data", os.ModePerm)
    err = neuralnetwork.Save(network, "./data/" + name + ".json")
    if (err != nil) {
        return err
    }

    return nil
}
