"use strict";

require("../src/feedforwardNeuralNetworks");
var Matrix = require('ml-matrix');

describe('Feedforward Neural Networks', function () {
    it('Training the neural network with XOR', function () {
        var X = Matrix([[0, 0], [0, 1], [1, 0], [1, 1]]);

        var labels = [0, 1, 1, 0];
        /*for(var i = 1e-5; i < 2; i *= 2) {
            for(var j = 1e-5; j < 2; j *= 2) {
                var xorNeuralNetwork = new FeedforwardNeuralNetworks();
                xorNeuralNetwork.train(X, labels, i, j, 2, 1500, 6);
                var result = xorNeuralNetwork.predict(Matrix([[1, 1], [0, 1], [1, 0], [0, 0]]));

                if(!(result[0][0] > result[0][1])) continue;
                if(result[1][0] > result[1][1]) continue;
                if(result[2][0] > result[2][1]) continue;
                if(!(result[3][0] > result[3][1])) continue;

                console.log("i: " + i + " j: " + j);
            }
        }*/

        var xorNeuralNetwork = new FeedforwardNeuralNetworks();
        xorNeuralNetwork.train(X, labels, 0.4, 0.001, 2, 23, 25);
        var result = xorNeuralNetwork.predict(Matrix([[1, 1], [1, 0], [0, 1], [0, 0]]));

        console.log(result);

        (result[0][0] > result[0][1]).should.be.equal(true);
        (result[1][0] < result[1][1]).should.be.equal(true);
        (result[2][0] < result[2][1]).should.be.equal(true);
        (result[3][0] > result[3][1]).should.be.equal(true);
    });
});