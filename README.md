# dunnlen
## Easy-to-Use Deep Learning Library
All neural networks are created and implemented from scratch using solely NumPy and Python modules. This library is created for my own educational growth, and I hope to make it usable for the rest of the DL noobs out there. Wish me luck on this creative endeavor.

### Why "DunnLen?"
"Dunn" is the first syllable of my surname, which has Scottish origins. As for the "len" part, try saying "learn" in a heavy British accent.

![logo](rsz_logo.png?raw=True)
## Current Features
1. A basic single-layer perceptron
2. A multi-layer perceptron classifer that takes as input user-defined NN architecture (layer-by-layer dimensions and activation functions).
    * Note: activation functions for MLP classifier are limited to sigmoid and reLu
    * Weights and biases are initialized as either zeros, randomly within parameters set by each layer dimension, or by He initialization (He et al., 2015). Maybe might integrate Xavier initialization on a later date

## Future Additions
1. Multi-layer perceptron regressor
2. ...Other endemic species form the zoo of neural networks.
3. A "NN_Architecture" object in which to conveniently store activation sequences and layer dimensions. Object will be fed into and parsed by a NeuralNetwork object.

### Performance Additions
1. Batch sizes
2. Regularization
3. Batch normalization
