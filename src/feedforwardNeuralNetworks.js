"use strict";
var Matrix = require("ml-matrix");

module.exports = FeedforwardNeuralNetworks;

function randomInitialzeTheta(labelsIn, labelsOut) {
    var epsilonRange = 1; // values around
    return Matrix.rand(labelsOut, labelsIn).mulS(2).mulS(epsilonRange).addS(-epsilonRange);
}

function sigmoid(value) {
    return 1.0 / (1 + Math.exp(-value));
}

function sigmoidGradient(value) {
    var sig = sigmoid(value);
    return sig * (1 - sig);
}

function FeedforwardNeuralNetworks(X, y, parameters) {
    if(!(this instanceof FeedforwardNeuralNetworks)) {
        return new FeedforwardNeuralNetworks();
    }

    this.X = X;
    this.y = y;
    this.numberOfLayers = parameters.length;
    this.sizes = parameters;
    this.buildNetwork();
}

FeedforwardNeuralNetworks.prototype.buildNetwork = function buildNetwork() {
    this.weights = [];
    this.biases = [];
    this.inputs = [];
    this.outputs = [];
    this.errors = [];

    for(var layers = 0; layers < this.numberOfLayers - 1; ++layers) {
        var n = this.sizes[layers];
        var m = this.sizes[layers + 1];
        this.weights.append(randomInitialzeTheta(n, m));
        this.biases.append(randomInitialzeTheta(m, 1));
        this.inputs.append(Matrix.zeros(n, 1));
        this.outputs.append(Matrix.zeros(n, 1));
        this.errors.append(Matrix.zeros(n, 1));
    }

    // the last one
    n = this.sizes[this.sizes.length - 1];
    this.inputs.append(Matrix.zeros(n, 1));
    this.outputs.append(Matrix.zeros(n, 1));
    this.errors.append(Matrix.zeros(n, 1));
};

FeedforwardNeuralNetworks.prototype.feedforward = function feedforward(X) {
    this.inputs[0] = X;
    this.outputs[0] = X;
    for(var i = 1; i < this.numberOfLayers; ++i) {
        this.inputs[i] = this.weights[i - 1].dot(this.outputs[i - 1]) + this.biases[i - 1];
        this.outputs[i] = sigmoid(this.inputs[i]);
    }
    return this.outputs[this.outputs.length - 1];
};

FeedforwardNeuralNetworks.prototype.updateWeights = function updateWeights(X, y) {
    var output = this.feedforward(X);
    var n = this.numberOfLayers - 1;
    this.errors[n] = sigmoid(this.outputs[n - 1])*(output - y);

    for(var i = n - 1; i > 0; --i) {
        this.errors[i] = sigmoidGradient(this.inputs[i]) * (this.weights[i].transpose()
                                                            .dot(this.errors[i + 1]));
        this.weights[i].sub(this.errors[i + 1].transpose()
                          .mmul(this.outputs[i]).mulS(this.learningRate));
        this.biases[i] = this.biases[i].sub( this.errors[i + 1].clone().mulS(this.learningRate) );
    }

    this.weights[0].sub(this.errors[1].transpose()
                            .mmul(this.outputs[0]).mulS( this.learningRate));
    this.biases[0].sub(this.errors[1].clone().mul(this.learningRate));
};

FeedforwardNeuralNetworks.prototype.train = function train(iterations, learningRate) {
    this.learningRate = learningRate;
    var n = this.X.rows;
    for(var i = 0; i < iterations; ++i) {
        for(var j = 0; j < n; ++i) {
            var x = this.X.getRow(j);
            var y = this.y[j];
            this.updateWeights(x, y);
        }
    }
};

FeedforwardNeuralNetworks.prototype.predict = function predict(x) {
    return this.feedforward(x);
};