# Feedforward Neural Network

A implementation of feedforward neural networks in javascript based on the mrbo answer
found here:

[Implementation] (http://stackoverflow.com/questions/9951487/implementing-a-neural-network-in-java-training-and-backpropagation-issues)

## Methods

### new FeedforwardNeuralNetwork([LayersSize])

__Arguments__

* `layersSize` - Array of numbers with sizes of each layer.

__Example__

```js
var FNN = new FeedforwardNeuralNetwork([2, 4, 1]);
```

### train(trainingSet, predictions, learningRate, momentum)

Train the Neural Network with a given training set, predictions, learning rate and a 
momentum (Regularization term).

__Arguments__

* `trainingSet` - A matrix of the training set.
* `predictions` - A matrix of predictions with the same size of rows of the trainingSet.
* `learningRate` - The learning rate (number).
* `momentum` - The regularization term (number).

__Example__

```js
var trainingSet = [[0, 0], [0, 1], [1, 0], [1, 1]];
var predictions = [[0], [0], [0], [1]];

FNN.train(trainingSet, predictions, 0.3, 0.3);
```

### predict(dataset)

Predict the values of the dataset.

__Arguments__

* `dataset` - A matrix that contains the dataset.

__Example__

```js
var dataset = [[0, 0], [0, 1], [1, 0], [1, 1]];

var ans = FNN.predict(dataset);
```

### export()

Exports the actual Neural Network to an Javascript Object.

### load(model)

Returns a new Neural Network with the given model.

__Arguments__

* `model` - Javascript Object generated from export() function.

## Authors

- [Jefferson Hernandez](https://github.com/JeffersonH44)

## License

[MIT](./LICENSE)