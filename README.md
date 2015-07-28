# Feedforward Neural Network

  [![NPM version][npm-image]][npm-url]
  [![build status][travis-image]][travis-url]
  [![David deps][david-image]][david-url]
  [![npm download][download-image]][download-url]

A implementation of feedforward neural networks in javascript based on the mrbo answer
found here:

[Implementation] (http://stackoverflow.com/questions/9951487/implementing-a-neural-network-in-java-training-and-backpropagation-issues)

## Methods

### new FeedforwardNeuralNetwork([LayersSize])

__Arguments__

* `layersSize` - Array of numbers with sizes of each layer.

__Example__

```js
var fnn = new FeedforwardNeuralNetwork();
```

### train(trainingSet, predictions, iterations, learningRate, momentum)

Train the Neural Network with a given training set, predictions, learning rate and a 
momentum (Regularization term).

__Arguments__

* `trainingSet` - A matrix of the training set.
* `predictions` - A matrix of predictions with the same size of rows of the trainingSet.
* `options` - A Javascript object with the configuration of the FNN.

__Options__

* `hiddenLayers` - Array with the size of each hidden layer in the FNN.
* `iterations` - Maximum number of iterations of the algorithm.
* `learningRate` - The learning rate (number).
* `momentum` - The regularization term (number).

__Example__

```js
var trainingSet = [[0, 0], [0, 1], [1, 0], [1, 1]];
var predictions = [[0], [0], [0], [1]];
var options = {
  hiddenLayers: [4],
  iterations: 100,
  learningRate: 0.3,
  momentum: 0.3
};

fnn.train(trainingSet, predictions, options);
```

### predict(dataset)

Predict the values of the dataset.

__Arguments__

* `dataset` - A matrix that contains the dataset.

__Example__

```js
var dataset = [[0, 0], [0, 1], [1, 0], [1, 1]];

var ans = fnn.predict(dataset);
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

[npm-image]: https://img.shields.io/npm/v/ml-fnn.svg?style=flat-square
[npm-url]: https://npmjs.org/package/ml-fnn
[travis-image]: https://img.shields.io/travis/mljs/feedforward-neural-networks/master.svg?style=flat-square
[travis-url]: https://travis-ci.org/mljs/feedforward-neural-networks
[david-image]: https://img.shields.io/david/mljs/feedforward-neural-networks.svg?style=flat-square
[david-url]: https://david-dm.org/mljs/feedforward-neural-networks
[download-image]: https://img.shields.io/npm/dm/ml-fnn.svg?style=flat-square
[download-url]: https://npmjs.org/package/ml-fnn