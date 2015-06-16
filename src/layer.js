/**
 * Created by jefferson on 6/16/15.
 */
var Matrix = require("ml-matrix");

module.exports = Layer;

function randomInitialzeWeights(numberOfWeights) {
    return Matrix.rand(1, numberOfWeights).sub(0.5).mul(4).getRow(0);
}

function sigmoid(value) {
    return 1.0 / (1 + Math.exp(-value));
}

function sigmoidGradient(value) {
    var sig = sigmoid(value);
    return sig * (1 - sig);
}

function Layer(inputSize, outputSize) {
    this.output = Matrix.zeros(1, outputSize).getRow(0);
    this.input = Matrix.zeros(1, inputSize + 1).getRow(0); //+1 for bias term
    this.deltaWeights = Matrix.zeros(1, (1 + inputSize) * outputSize).getRow(0);
    this.weights = randomInitialzeWeights(this.deltaWeights.length);
    this.isSigmoid = true;
}

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

Layer.prototype.train = function (error, learningRate, momentum) {
    var offs = 0;
    var nextError = Matrix.zeros(1, this.input.length).getRow(0);//new Array(this.input.length);

    for(var i = 0; i < this.output.length; ++i) {
        var delta = error[i];
        delta *= this.isSigmoid ?  sigmoidGradient(this.output[i]) : 1;

        for(var j = 0; j < this.input.length; ++j) {
            var index = offs + j;
            nextError[i] += this.weights[index] * delta;

            var deltaWeight = this.input[j] * delta * learningRate;
            this.weights[index] += this.deltaWeights[index] * momentum + deltaWeight;
            this.deltaWeights[index] = deltaWeight;
        }

        offs += this.input.length;
    }

    return nextError;
};