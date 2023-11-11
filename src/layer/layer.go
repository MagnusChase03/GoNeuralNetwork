// The layer package is used to define
// the structure of neural network layers
// and how they work
package layer

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