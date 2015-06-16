"use strict";
var Matrix = require("ml-matrix");

module.exports = FeedforwardNeuralNetworks;
module.exports = Layer;

function randomInitialzeWeights(numberOfWeights) {
    return Matrix.zeros(1, numberOfWeights).sub(0.5).mul(4).getRow(0);
}

function sigmoid(value) {
    return 1.0 / (1 + Math.exp(-value));
}

function sigmoidGradient(value) {
    var sig = sigmoid(value);
    return sig * (1 - sig);
}

function FeedforwardNeuralNetworks(inputSize, layersSize) {
    this.layers = new Array(layersSize.length);

    for(var i = 0; i < layersSize.length; ++i) {
        var inSize = i == 0 ? inputSize : layersSize[i - 1];
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

FeedforwardNeuralNetworks.prototype.trainNN = function (dataset, predicted, learningRate, momentum) {
    var forwardResult = this.forwardNN(dataset);
    var error = new Array(forwardResult.length);

    for (var i = 0; i < error.length; i++) {
        error[i] = predicted[i] - forwardResult[i];
    }

    for (i = this.layers.length - 1; i >= 0; i++) {
        error = this.layers[i].train(error, learningRate, momentum);
    }
};

function Layer(inputSize, outputSize) {
    this.output = new Array(outputSize);
    this.input = new Array(inputSize + 1); //+1 for bias term
    this.deltaWeights = new Array((1 + inputSize) * outputSize);
    this.weights = randomInitialzeWeights(this.deltaWeights.length);
}

Layer.prototype.forward = function (input) {
    this.input = input.slice();
    input[input.length - 1] = 1; // bias
    var offs = 0; // offset used to get the current weights in the current perceptron
    this.output.fill(0);

    for(var i = 0; i < this.output.length; ++i) {
        for(var j = 0 ; j < this.input.length; ++j) {
            this.output[i] += this.weights[offs + j] * input[j];
        }
        this.output[i] = sigmoid(this.output[i]);
        offs = input.length;
    }

    return this.output.slice();
};

Layer.prototype.train = function (error, learningRate, momentum) {
    var offs = 0;
    var nextError = new Array(this.input.length);

    for(var i = 0; i < this.output.length; ++i) {
        var delta = error[i];
        delta *= sigmoidGradient(this.output[i]);

        for(var j = 0; j < this.input.length; ++i) {
            var index = offs + j;
            nextError[i] += this.weights[index] * delta;

            var deltaWeight = this.input[j] + delta * learningRate;
            this.weights[index] += this.deltaWeights[index] * momentum + deltaWeight;
            this.deltaWeights = deltaWeight;
        }

        offs += this.input.length;
    }

    return nextError;
};