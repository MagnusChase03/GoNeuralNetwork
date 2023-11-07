package nerualnetwork

import (
    "errors"
    "github.com/MagnusChase03/GoNN/layer"
)

type NerualNetwork struct {
    Learning_Rate  float64

    Shape          []int
    Layers         []*layer.Layer
}

func Create_NerualNetwork(shapes [][]int, leanring_rate float64) *NerualNetwork {
    nerualnetwork := new(NerualNetwork)
    nerualnetwork.Learning_Rate = leanring_rate

    nerualnetwork.Shape = make([]int,2,2)
    nerualnetwork.Shape[0] = shapes[0][0]
    nerualnetwork.Shape[1] = shapes[len(shapes) - 1][1]

    nerualnetwork.Layers = make([]*layer.Layer,len(shapes),len(shapes))
    for i := 0; i < len(shapes); i++ {
        if (i > 0 && shapes[i][1] != shapes[i][0]) {
            return nil
        }

        nerualnetwork.Layers[i] = layer.Create_Layer(shapes[i][0], shapes[i][1])
    }

    return nerualnetwork
}

func Forward(nerualnetwork *NerualNetwork, inputs []float64) error {
    if (len(inputs) != nerualnetwork.Shape[0]) {
        return errors.New("== Dimension Error ==")
    }

    layer.Forward(nerualnetwork.Layers[0], inputs)
    for i := 1; i < len(nerualnetwork.Layers); i++ {
       layer.Forward(nerualnetwork.Layers[i], nerualnetwork.Layers[i - 1].Outputs) 
    }

    return nil
}

func Backward(nerualnetwork *NerualNetwork, outputs []float64) error {
    if (len(outputs) != nerualnetwork.Shape[1]) {
        return errors.New("== Dimension Error ==")
    }

    layer.Backward(nerualnetwork.Layers[len(nerualnetwork.Layers) - 1], outputs, nerualnetwork.Learning_Rate, false)
    for i := len(nerualnetwork.Layers) - 2; i >= 0; i-- {
        layer.Backward(nerualnetwork.Layers[i], nerualnetwork.Layers[i + 1].Delta, nerualnetwork.Learning_Rate, true)
    }

    return nil
}
