"use strict";

var Layer = require("./layer");
var Matrix = require("ml-matrix");

module.exports = FeedforwardNeuralNetwork;

/**
 * Function that returns a random number between two numbers (inclusive)
 * @param {number} min - lower bound
 * @param {number} max - upper bound.
 * @returns {number} random number
 */
function randomIntegerFromInterval(min, max) {
    return Math.floor(Math.random()*(max - min + 1) + min);
}

/**
 * Constructor for the FNN (Feedforward Neural Networks) that takes an Array of Numbers,
 * those numbers corresponds to the size of each layer in the FNN, the first and the last number of the array corresponds to the input and the
 * output layer respectively.
 *
 * @param reload - for load purposes.
 * @param model - for load purposes.
 * @constructor
 */
function FeedforwardNeuralNetwork(reload, model) {
    if(reload) {
        this.layers = model.layers;
        this.inputSize = model.inputSize;
        this.outputSize = model.outputSize;
    }
}

/**
 * Build the Neural Network with an array that represent each hidden layer size.
 *
 * @param {Array} layersSize - Array of sizes of each layer.
 */
FeedforwardNeuralNetwork.prototype.buildNetwork = function (layersSize) {
    layersSize.push(this.outputSize);

    this.layers = new Array(layersSize.length);

    for (var i = 0; i < layersSize.length; ++i) {
        var inSize = (i == 0) ? this.inputSize : layersSize[i - 1];
        this.layers[i] = new Layer(inSize, layersSize[i]);
    }

    this.layers[this.layers.length - 1].isSigmoid = false;
};

/**
 * Function that applies a forward propagation over the Neural Network
 * with one case of the dataset.
 * @param {Array} input - case of the dataset.
 * @returns {Array} result of the forward propagation.
 */
FeedforwardNeuralNetwork.prototype.forwardNN = function (input) {
    var results = input.slice();

    for(var i = 0; i < this.layers.length; ++i) {
        results = this.layers[i].forward(results);
    }

    return results;
};

/**
 * Function that makes one iteration (epoch) over the Neural Network with one element
 * of the dataset with corresponding prediction; the other two arguments are the
 * learning rate and the momentum that is the regularization term for the parameters
 * of each perceptron in the Neural Network.
 * @param {Array} data - Element of the dataset.
 * @param {Array} prediction - Prediction over the data object.
 * @param {Number} learningRate
 * @param momentum - the regularization term.
 */
FeedforwardNeuralNetwork.prototype.iteration = function (data, prediction, learningRate, momentum) {
    var forwardResult = this.forwardNN(data);
    var error = new Array(forwardResult.length);

    if(typeof(prediction) === 'number')
        prediction = [prediction];

    for (var i = 0; i < error.length; i++) {
        error[i] = prediction[i] - forwardResult[i];
    }

    var lengthLayers = this.layers.length;

    for(i = 0; i < lengthLayers; ++i) {
        error = this.layers[lengthLayers - 1 - i].train(error, learningRate, momentum);
    }
};

/**
 * Method that train the neural network with a given training set with corresponding
 * predictions. The options argument has an array of the number of perceptrons that we want in each hidden layer, the
 * number of iterations (default 50) that we want to perform, the learning rate and the momentum that is the
 * regularization term (default 0.1 for both) for the parameters of each perceptron in the Neural Network.
 *
 * options:
 * * hiddenLayers - Array of number with each hidden layer size.
 * * iterations - Number
 * * learningRate - Number
 * * momentum - Number
 *
 * @param {Matrix} trainingSet
 * @param {Matrix} predictions
 * @param {Number} options
 */
FeedforwardNeuralNetwork.prototype.train = function (trainingSet, predictions, options) {
    if(options === undefined) options = {};

    if(trainingSet.length !== predictions.length)
        throw new RangeError("the training and prediction set must have the same size.");

    this.inputSize = trainingSet[0].length;
    this.outputSize = predictions[0].length;

    var hiddenLayers = options.hiddenLayers === undefined ? [10] : options.hiddenLayers;
    var iterations = options.iterations === undefined ? 50 : options.iterations;
    var learningRate = options.learningRate === undefined ? 0.1 : options.learningRate;
    var momentum = options.momentum === undefined ? 0.1 : options.momentum;

    this.buildNetwork(options.hiddenLayers);

    for(var i = 0; i < iterations; ++i) {
        for(var j = 0; j < predictions.length; ++j) {
            var index = randomIntegerFromInterval(0, predictions.length - 1);
            this.iteration(trainingSet[index], predictions[index], learningRate, momentum);
        }
    }
};

/**
 * Function that with a dataset, gives all the predictions for this dataset.
 * @param {Matrix} dataset.
 * @returns {Array} predictions
 */
FeedforwardNeuralNetwork.prototype.predict = function (dataset) {
    if(dataset[0].length !== this.inputSize)
        throw new RangeError("The dataset columns must have the same size of the " +
                             "input layer");
    var result = new Array(dataset.length);
    for (var i = 0; i < dataset.length; i++) {
        result[i] = this.forwardNN(dataset[i]);
    }

    result = Matrix(result);
    return result.columns === 1 ? result.getColumn(0) : result;
};

/**
 * function that loads a object model into the Neural Network.
 * @param model
 * @returns {FeedforwardNeuralNetwork} with the provided model.
 */
FeedforwardNeuralNetwork.load = function (model) {
    if(model.modelName !== "FNN")
        throw new RangeError("The given model is invalid!");

    return new FeedforwardNeuralNetwork(true, model);
};

/**
 * Function that exports the actual Neural Network to an object.
 * @returns {{modelName: string, layers: *, inputSize: *, outputSize: *}}
 */
FeedforwardNeuralNetwork.prototype.export = function () {
    return {
        modelName: "FNN",
        layers: this.layers,
        inputSize: this.inputSize,
        outputSize: this.outputSize
    };
};