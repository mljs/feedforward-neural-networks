"use strict";

var FeedforwardNeuralNetwork =require("../src/feedforwardNeuralNetwork");

describe('Feedforward Neural Networks', function () {

    /*
     * Dear user:
     *
     * Be careful when you are testing this module because this is
     * a Neural Network, sometimes doesn't converge but if you try
     * 3 or 4 times you will see that this module predict the XOR
     *
     * */
    it('Training the neural network with XOR', function () {
        var trainingSet = [[0, 0], [0, 1], [1, 0], [1, 1]];
        var predictions = [0, 1, 1, 0];

        var xorNN = new FeedforwardNeuralNetwork(2, [2, 1]);
        xorNN.layers[1].isSigmoid = false;
        xorNN.train(trainingSet, predictions, 500, 0.3, 0.3);
        xorNN.forwardNN([0, 0]).should.be.approximately(0, 3e-1);
        xorNN.forwardNN([0, 1]).should.be.approximately(1, 3e-1);
        xorNN.forwardNN([1, 0]).should.be.approximately(1, 3e-1);
        xorNN.forwardNN([1, 1]).should.be.approximately(0, 3e-1);

    });
});