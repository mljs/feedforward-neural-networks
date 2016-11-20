# Feedforward Neural Network

  [![NPM version][npm-image]][npm-url]
  [![build status][travis-image]][travis-url]
  [![David deps][david-image]][david-url]
  [![npm download][download-image]][download-url]

A implementation of feedforward neural networks in javascript based on the mrbo answer
found here:

[Implementation] (http://stackoverflow.com/questions/9951487/implementing-a-neural-network-in-java-training-and-backpropagation-issues)

## Methods

### new FNN(X, Y)

### train(options)

__Options__

* `hiddenLayers` - Array with the size of each hidden layer in the FNN.
* `hiddenOptions` - (optional) Array with the options for each layer of the FNN specifiying activating functions of the form `{nonLinearity:'sigmoid' or 'tanh'}`.
* `iterations` - Maximum number of iterations of the algorithm.
* `learningRate` - The learning rate (number).
* `momentum` - The regularization term (number).

### predict(dataset)

### toJSON()

### FNN.load(model)

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
