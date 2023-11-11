// The activation functions package
// contains different functions that
// are used to non linearize data
// within the neural network
package activationfunctions

import (
	"math"
)

// The ActivationFunction interface defines
// the methods required of an activation
// function.
type ActivationFunction interface {
	Calculate(x float64)   float64
	Derivative(x float64)  float64
}

// Sigmoid is an activation function.
type Sigmoid struct {}

// Implements the calculate function for
// sigmoid.
func (s Sigmoid) Calculate(x float64) float64 {
    return 1 / (1 + math.Exp(-x))
}

// Implements the derivative function for
// sigmoid.
func (s Sigmoid) Derivative(x float64) float64 {
	return s.Calculate(x) * (1 - s.Calculate(x))
}