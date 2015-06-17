"use strict";

var Layer = require("./layer");
var Matrix = require("ml-matrix");

module.exports = FeedforwardNeuralNetworks;

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
 * Constructor for the FNN (Feedforward Neural Networks) that takes the input size and
 * an array of the size of each hidden layer, the last element must be the size of the
 * output of the model.
 * @param {Number} inputSize -
 * @param layersSize
 * @param reload - for load purposes.
 * @param model - for load purposes.
 * @constructor
 */
function FeedforwardNeuralNetworks(layersSize, reload, model) {
    if(reload) {
        this.layers = model.layers;
        this.inputSize = model.inputSize;
        this.outputSize = model.outputSize;
    } else {
        this.inputSize = layersSize[0];
        this.outputSize = layersSize[layersSize.length - 1];
        layersSize.shift();

        this.layers = new Array(layersSize.length);

        for (var i = 0; i < layersSize.length; ++i) {
            var inSize = (i == 0) ? this.inputSize : layersSize[i - 1];
            this.layers[i] = new Layer(inSize, layersSize[i]);
        }

        this.layers[this.layers.length - 1].isSigmoid = false;
    }
}

FeedforwardNeuralNetworks.prototype.forwardNN = function (input) {
    var results = input.slice();

    for(var i = 0; i < this.layers.length; ++i) {
        results = this.layers[i].forward(results);
    }

    return results;
};

FeedforwardNeuralNetworks.prototype.iteration = function (dataset, predicted, learningRate, momentum) {
    var forwardResult = this.forwardNN(dataset);
    var error = new Array(forwardResult.length);

    if(typeof(predicted) === 'number')
        predicted = [predicted];

    for (var i = 0; i < error.length; i++) {
        error[i] = predicted[i] - forwardResult[i];
    }

    for (i = this.layers.length - 1; i >= 0; i--) {
        error = this.layers[i].train(error, learningRate, momentum);
    }
};

FeedforwardNeuralNetworks.prototype.train = function (trainingSet, predictions, iterations, learningRate, momentum) {

    for(var i = 0; i < iterations; ++i) {
        for(var j = 0; j < predictions.length; ++j) {
            var index = randomIntegerFromInterval(0, predictions.length - 1);
            this.iteration(trainingSet[index], predictions[index], learningRate, momentum);
        }
    }
};

FeedforwardNeuralNetworks.prototype.predict = function (dataset) {
    var result = new Array(dataset.length);
    for (var i = 0; i < dataset.length; i++) {
        result[i] = this.forwardNN(dataset[i]);
    }

    result = Matrix(result);
    return result.columns === 1 ? result.getColumn(0) : result;
};

FeedforwardNeuralNetworks.load = function (model) {
    if(model.modelName !== "FNN")
        throw new RangeError("The given model is invalid!");

    return new FeedforwardNeuralNetworks(null, null, true, model);
};

FeedforwardNeuralNetworks.prototype.export = function () {
    return {
        modelName: "FNN",
        layers: this.layers,
        inputSize: this.inputSize,
        outputSize: this.outputSize
    };
};