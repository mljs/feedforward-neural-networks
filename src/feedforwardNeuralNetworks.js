"use strict";
var Layer = require("./layer");

module.exports = FeedforwardNeuralNetworks;

function randomIntegerFromInterval(min, max) {
    return Math.floor(Math.random()*(max - min + 1) + min);
}

function FeedforwardNeuralNetworks(inputSize, layersSize) {
    this.layers = new Array(layersSize.length);

    for(var i = 0; i < layersSize.length; ++i) {
        var inSize = (i == 0) ? inputSize : layersSize[i - 1];
        this.layers[i] = new Layer(inSize, layersSize[i]);
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