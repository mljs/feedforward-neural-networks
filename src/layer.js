"use strict";

var Matrix = require("ml-matrix");

module.exports = Layer;

/**
 * Function that create a random array of numbers between -2 to 2.
 * @param numberOfWeights - size of the array.
 * @returns {Array} random array of numbers.
 */
function randomInitialzeWeights(numberOfWeights) {
    return Matrix.rand(1, numberOfWeights).sub(0.5).mul(4).getRow(0);
}

/**
 * Function that calculates the sigmoid (logistic) function.
 * @param value
 * @returns {number}
 */
function sigmoid(value) {
    return 1.0 / (1 + Math.exp(-value));
}

/**
 * Function that calculates the derivate of the sigmoid function.
 * @param value
 * @returns {number}
 */
function sigmoidGradient(value) {
    return value * (1 - value);
}

/**
 * Constructor that creates a layer for the neural network given the number of inputs
 * and outputs.
 * @param inputSize
 * @param outputSize
 * @constructor
 */
function Layer(inputSize, outputSize) {
    this.output = Matrix.zeros(1, outputSize).getRow(0);
    this.input = Matrix.zeros(1, inputSize + 1).getRow(0); //+1 for bias term
    this.deltaWeights = Matrix.zeros(1, (1 + inputSize) * outputSize).getRow(0);
    this.weights = randomInitialzeWeights(this.deltaWeights.length);
    this.isSigmoid = true;
}

/**
 * Function that performs the forward propagation for the current layer
 * @param {Array} input - output from the previous layer.
 * @returns {Array} output - output for the next layer.
 */
Layer.prototype.forward = function (input) {
    this.input = input.slice();
    this.input.push(1); // bias
    var offs = 0; // offset used to get the current weights in the current perceptron
    this.output = Matrix.zeros(1, this.output.length).getRow(0);

    for(var i = 0; i < this.output.length; ++i) {
        for(var j = 0 ; j < this.input.length; ++j) {
            this.output[i] += this.weights[offs + j] * this.input[j];
        }
        if(this.isSigmoid)
            this.output[i] = sigmoid(this.output[i]);

        offs += this.input.length;
    }

    return this.output.slice();
};

/**
 * Function that performs the backpropagation algorithm for the current layer.
 * @param {Array} error - errors from the previous layer.
 * @param {Number} learningRate - Learning rate for the actual layer.
 * @param {Number} momentum - The regularizarion term.
 * @returns {Array} the error for the next layer.
 */
Layer.prototype.train = function (error, learningRate, momentum) {
    var offs = 0;
    var nextError = Matrix.zeros(1, this.input.length).getRow(0);//new Array(this.input.length);

    for(var i = 0; i < this.output.length; ++i) {
        var delta = error[i];

        if(this.isSigmoid)
            delta *= sigmoidGradient(this.output[i]);

        for(var j = 0; j < this.input.length; ++j) {
            var index = offs + j;
            nextError[j] += this.weights[index] * delta;

            var deltaWeight = this.input[j] * delta * learningRate;
            this.weights[index] += this.deltaWeights[index] * momentum + deltaWeight;
            this.deltaWeights[index] = deltaWeight;
        }

        offs += this.input.length;
    }

    return nextError;
};