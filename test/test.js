"use strict";

var FeedforwardNeuralNetwork =require("../src/feedforwardNeuralNetwork");

describe('Feedforward Neural Networks', function () {

    it('Training the neural network with XOR operator', function () {
        var trainingSet = [[0, 0], [0, 1], [1, 0], [1, 1]];
        var predictions = [[0], [1], [1], [0]];

        var xorNN = new FeedforwardNeuralNetwork([2, 4, 1]);
        xorNN.train(trainingSet, predictions, 500, 0.3, 0.3);
        var results = xorNN.predict(trainingSet);

        (results[0]).should.be.approximately(predictions[0], 3e-1);
        (results[1]).should.be.approximately(predictions[1], 3e-1);
        (results[2]).should.be.approximately(predictions[2], 3e-1);
        (results[3]).should.be.approximately(predictions[3], 3e-1);
    });

    it('Training the neural network with AND operator', function () {
        var trainingSet = [[0, 0], [0, 1], [1, 0], [1, 1]];
        var predictions = [[1, 0], [1, 0], [1, 0], [0, 1]];

        var andNN = new FeedforwardNeuralNetwork([2, 3, 2]);
        andNN.train(trainingSet, predictions, 500, 0.3, 0.3);

        var results = andNN.predict(trainingSet);

        (results[0][0] > results[0][1]).should.be.ok;
        (results[1][0] > results[1][1]).should.be.ok;
        (results[2][0] > results[2][1]).should.be.ok;
        (results[3][0] < results[3][1]).should.be.ok;
    });

    it('Export and import', function () {
        var trainingSet = [[0, 0], [0, 1], [1, 0], [1, 1]];
        var predictions = [[0], [1], [1], [1]];

        var orNN = new FeedforwardNeuralNetwork([2, 4, 1]);
        orNN.train(trainingSet, predictions, 500, 0.3, 0.3);

        var model = orNN.export();
        var neworNN = FeedforwardNeuralNetwork.load(model);

        var results = neworNN.predict(trainingSet);

        (results[0]).should.be.approximately(predictions[0], 3e-1);
        (results[1]).should.be.approximately(predictions[1], 3e-1);
        (results[2]).should.be.approximately(predictions[2], 3e-1);
        (results[3]).should.be.approximately(predictions[3], 3e-1);
    });
});