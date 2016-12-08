'use strict';

var FeedforwardNeuralNetwork = require('..');
var Matrix = require('ml-matrix');

describe('Feedforward Neural Networks using sigmoid nonlinearity', function () {

    it('Training the neural network with XOR operator', function () {
        let trainingSet = new Matrix([[0, 0], [0, 1], [1, 0], [1, 1]]);
        let predictions = [0, 1, 1, 0];

        let options = {
            hiddenLayers: [4],
            iterations: 1000,
            learningRate: 0.3
        };
        let xorNN = new FeedforwardNeuralNetwork(options);

        xorNN.train(trainingSet, predictions);
        let results = xorNN.predict(trainingSet);

        results[0].should.be.equal(predictions[0]);
        results[1].should.be.equal(predictions[1]);
        results[2].should.be.equal(predictions[2]);
        results[3].should.be.equal(predictions[3]);
    });

    it('Training the neural network with AND operator', function () {
        let trainingSet = [[0, 0], [0, 1], [1, 0], [1, 1]];
        let predictions = [[1, 0], [1, 0], [1, 0], [0, 1]];

        let options = {
            hiddenLayers: [3],
            iterations: 500,
            learningRate: 0.3
        };
        let andNN = new FeedforwardNeuralNetwork(options);
        andNN.train(trainingSet, predictions);

        let results = andNN.predict(trainingSet);

        (results[0][0]).should.be.greaterThan(results[0][1]);
        (results[1][0]).should.be.greaterThan(results[1][1]);
        (results[2][0]).should.be.greaterThan(results[2][1]);
        (results[3][0]).should.be.lessThan(results[3][1]);
    });

    it('Export and import', function () {
        let trainingSet = [[0, 0], [0, 1], [1, 0], [1, 1]];
        let predictions = [0, 1, 1, 1];

        let options = {
            hiddenLayers: [4],
            iterations: 500,
            learningRate: 0.3
        };
        let orNN = new FeedforwardNeuralNetwork(options);
        orNN.train(trainingSet, predictions);

        let model = JSON.parse(JSON.stringify(orNN));
        let networkNN = FeedforwardNeuralNetwork.load(model);

        let results = networkNN .predict(trainingSet);

        (results[0]).should.be.approximately(predictions[0], 3e-1);
        (results[1]).should.be.approximately(predictions[1], 3e-1);
        (results[2]).should.be.approximately(predictions[2], 3e-1);
        (results[3]).should.be.approximately(predictions[3], 3e-1);
    });

    it('Multiclass clasification', function () {
        let trainingSet = [[0, 0], [0, 1], [1, 0], [1, 1]];
        let predictions = [2, 0, 1, 0];

        let options = {
            hiddenLayers: [4],
            iterations: 300,
            learningRate: 0.5
        };
        let nn = new FeedforwardNeuralNetwork(options);
        nn.train(trainingSet, predictions);

        let result = nn.predict(trainingSet);

        result[0].should.be.approximately(2, 1e-1);
        result[1].should.be.approximately(0, 1e-1);
        result[2].should.be.approximately(1, 1e-1);
        result[3].should.be.approximately(0, 1e-1);
    });

    it('Big case', function () {
        let trainingSet = [[1, 1], [1, 2], [2, 1], [2, 2], [3, 1], [1, 3], [1, 4], [4, 1],
                            [6, 1], [6, 2], [6, 3], [6, 4], [6, 5], [5, 5], [4, 5], [3, 5]];
        let predictions = [[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0],
                            [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]];

        let options = {
            hiddenLayers: [10],
            iterations: 200,
            learningRate: 0.1
        };
        let nn = new FeedforwardNeuralNetwork(options);
        nn.train(trainingSet, predictions);

        let result = nn.predict([[5, 4]]);

        result[0][0].should.be.lessThan(result[0][1]);
    });
});
