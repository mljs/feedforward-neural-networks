"use strict";

var Matrix = require("ml-matrix");

class Layer {
    /**
     * Constructor that creates a layer for the neural network given the number of inputs
     * and outputs.
     * @param inputSize
     * @param outputSize
     * @param options: a dictionary of parameters for the layer
     *               {nonLinearity : 'sigmoid'} 
     * @constructor
     */
    constructor(inputSize, outputSize, options) {
        
        options = options || {nonLinearity: 'sigmoid'};

        this.output = Matrix.zeros(1, outputSize).getRow(0);
        this.input = Matrix.zeros(1, inputSize + 1).getRow(0); //+1 for bias term
        this.deltaWeights = Matrix.zeros(1, (1 + inputSize) * outputSize).getRow(0);
        this.weights = randomInitializeWeights(this.deltaWeights.length, inputSize, outputSize);
        
        if (!ACTIVATION.hasOwnProperty(options.nonLinearity)) {
            throw ReferenceError('No activation function named' + options.nonLinearity);
        }
        this.activation = ACTIVATION[options.nonLinearity];
        this.activationGradient = ACTIVATION_GRADIENT[options.nonLinearity];
        this.nonLinearity = options.nonLinearity;
        this.stopActivation = false;
    }

    /**
     * Function that performs the forward propagation for the current layer
     * @param {Array} input - output from the previous layer.
     * @returns {Array} output - output for the next layer.
     */
    forward(input) {
        this.input = input.slice();
        this.input.push(1); // bias
        var offs = 0; // offset used to get the current weights in the current perceptron
        this.output = Matrix.zeros(1, this.output.length).getRow(0);

        for (var i = 0; i < this.output.length; ++i) {
            for (var j = 0; j < this.input.length; ++j) {
                this.output[i] += this.weights[offs + j] * this.input[j];
            }

            if (!this.stopActivation) {
                this.output[i] = this.activation(this.output[i]);
            }

            offs += this.input.length;
        }

        return this.output.slice();
    }

    /**
     * Function that performs the backpropagation algorithm for the current layer.
     * @param {Array} error - errors from the previous layer.
     * @param {Number} learningRate - Learning rate for the actual layer.
     * @param {Number} momentum - The regularizarion term.
     * @returns {Array} the error for the next layer.
     */
    train(error, learningRate, momentum) {
        var offs = 0;
        var nextError = Matrix.zeros(1, this.input.length).getRow(0);//new Array(this.input.length);

        for (var i = 0; i < this.output.length; ++i) {
            var delta = error[i];

            if (!this.stopActivation) {
                delta *= this.activationGradient(this.output[i]);    
            }
            

            for (var j = 0; j < this.input.length; ++j) {
                var index = offs + j;
                nextError[j] += this.weights[index] * delta;

                var deltaWeight = this.input[j] * delta * learningRate;
                this.weights[index] += this.deltaWeights[index] * momentum + deltaWeight;
                this.deltaWeights[index] = deltaWeight;
            }

            offs += this.input.length;
        }

        return nextError;
    }
}

module.exports = Layer;

/**
 * Function that create a random array of numbers between value depending
 * on the input and output size given the following formula:
 *
 *    sqrt(6) / sqrt(l_in + l_out);
 *
 * Taken from the coursera course of machine learning from Andrew Ng,
 * Exercise 4, Page 7 of the exercise PDF.
 *
 * @param numberOfWeights - size of the array.
 * @param inputSize - number of input of the current layer
 * @param outputSize - number of output of the current layer
 * @returns {Array} random array of numbers.
 */
function randomInitializeWeights(numberOfWeights, inputSize, outputSize) {
    var epsilon = 2.449489742783 / Math.sqrt(inputSize + outputSize);
    return Matrix.rand(1, numberOfWeights).mul(2 * epsilon).sub(epsilon).getRow(0);
}

/**
 * Function that calculates the sigmoid (logistic) function at some value
 * @param value
 * @returns {number}
 */
function sigmoid(value) {
    return 1.0 / (1 + Math.exp(-value));
}

/**
 * Function that calculates the derivate of the sigmoid function.
 * given the value of the sigmoid function at that point
 * @param value
 * @returns {number}
 */
function sigmoidGradient(value) {
    return value * (1 - value);
}

/**
 * Function that caclulates the hyperbolic tangent (tanh) function at some value
 * @param value
 * @returns {number}
**/

function tanh(value) {
    return Math.tanh(value);
}

/**
 * Function that caclulates the derivative of 
 * hyperbolic tangent (tanh) function given the value
 * of the hyperbolic tangent function at that point
 * @param value
 * @returns {number}
**/


function tanhGradient(value) {
    return 1 - Math.pow(value, 2);
}

/**
 * Function that caclulates the rectified linear unit (RELU) function at some value
 * @param value
 * @returns {number}
**/

function relu(value) {
    if (value < 0) {
        return 0;
    } else {
        return value;
    }
}

/**
 * Function that caclulates the derivative of 
 * RELU function at some RELU value
 * @param value
 * @returns {number}
**/

function reluGradient(value) {
    if (value === 0) {
        return 0;
    } else {
        return 1;
    }
}

/**
 * Function that caclulates the leaky rectified linear unit (leaky RELU) 
 * function at some value
 * @param value
 * @returns {number}
**/

function leakyRelu(value) {
    if (value < 0) {
        return 0.001 * value;
    } else {
        return value;
    }
}

/**
 * Function that caclulates the derivative of 
 * leaky RELU function at some leaky RELU value
 * @param value
 * @returns {number}
**/

function leakyReluGradient(value) {
    if (value < 0) { // todo, this needs to be fixed.
        return 0.001;
    } else {
        return 1;
    }
}


/**
 * A dictionary that contains inbuilt activation functions
**/
var ACTIVATION = {
    'sigmoid': sigmoid,
    'tanh': tanh,
    'relu': relu,
    'leakyRelu': leakyRelu
};

/**
 * A dictionary that contains inbuild activation function gradients
 * Note: these must be evaluated with the value of the activation function
**/
var ACTIVATION_GRADIENT = {
    'sigmoid': sigmoidGradient,
    'tanh': tanhGradient,
    'relu': reluGradient,
    'leakyRelu': leakyReluGradient
};
