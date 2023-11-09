package main

import (
	"log"

	"github.com/MagnusChase03/gonn/neuralnetwork"	
)

func main() {
	//n, err := neuralnetwork.CreateNeuralNetwork([][]int{{3, 2}, {2, 2}}, 0.1)
	n, err := neuralnetwork.Load("tmp.dat")
	if err != nil {
		log.Fatalf("[ERROR] %w\n", err);
		return
	}

	for i := 0; i < 1000; i++ {
		if err := n.Forward([]float64{1.0, 2.0, 3.0}); err != nil {
			log.Fatalf("[ERROR] %w\n", err);
			return
		}

		if err := n.Backward([]float64{1.0, 0.0}); err != nil {
			log.Fatalf("[ERROR] %w\n", err);
			return
		}

		n.Update()

	}

	if err := n.Save("tmp.dat"); err != nil {
		log.Fatalf("[ERROR] %w\n", err);
		return
	}

}