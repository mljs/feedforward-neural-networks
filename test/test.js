"use strict";

require("../src/feedforwardNeuralNetworks");
var Matrix = require('ml-matrix');

describe('Feedforward Neural Networks', function() {
    it('Training the neural network with XOR', function () {
        var X = Matrix([[0, 0], [0, 1], [1, 0], [1, 1]]);

        var labels = [0, 1, 1, 0];

        var xorNeuralNetwork = new FeedforwardNeuralNetworks();
        xorNeuralNetwork.train(X, labels, 0.01, 0.0, 2, 20, 6);
        xorNeuralNetwork.predict([0, 0], [1, 0]);
    });
});