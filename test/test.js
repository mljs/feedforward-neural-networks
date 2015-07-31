"use strict";

var FeedforwardNeuralNetwork = require("..");

describe('Feedforward Neural Networks', function () {

    it('Training the neural network with XOR operator', function () {
        var trainingSet = [[0, 0], [0, 1], [1, 0], [1, 1]];
        var predictions = [[0], [1], [1], [0]];

        var xorNN = new FeedforwardNeuralNetwork();
        var options = {
            hiddenLayers: [4],
            iterations: 500,
            learningRate : 0.4,
            momentum: 0
        };

        xorNN.train(trainingSet, predictions, options);
        var results = xorNN.predict(trainingSet);

        (results[0]).should.be.approximately(predictions[0], 3e-1);
        (results[1]).should.be.approximately(predictions[1], 3e-1);
        (results[2]).should.be.approximately(predictions[2], 3e-1);
        (results[3]).should.be.approximately(predictions[3], 3e-1);
    });

    it('Training the neural network with AND operator', function () {
        var trainingSet = [[0, 0], [0, 1], [1, 0], [1, 1]];
        var predictions = [[1, 0], [1, 0], [1, 0], [0, 1]];

        var andNN = new FeedforwardNeuralNetwork();
        var options = {
            hiddenLayers: [3],
            iterations: 500,
            learningRate : 0.3,
            momentum: 0.3
        };
        andNN.train(trainingSet, predictions, options);

        var results = andNN.predict(trainingSet);

        (results[0][0]).should.be.greaterThan(results[0][1]);
        (results[1][0]).should.be.greaterThan(results[1][1]);
        (results[2][0]).should.be.greaterThan(results[2][1]);
        (results[3][0]).should.be.lessThan(results[3][1]);
    });

    it('Export and import', function () {
        var trainingSet = [[0, 0], [0, 1], [1, 0], [1, 1]];
        var predictions = [[0], [1], [1], [1]];

        var orNN = new FeedforwardNeuralNetwork();
        var options = {
            hiddenLayers: [4],
            iterations: 500,
            learningRate : 0.3,
            momentum: 0.3
        };
        orNN.train(trainingSet, predictions, options);

        var model = orNN.export();
        var neworNN = FeedforwardNeuralNetwork.load(model);

        var results = neworNN.predict(trainingSet);

        (results[0]).should.be.approximately(predictions[0], 3e-1);
        (results[1]).should.be.approximately(predictions[1], 3e-1);
        (results[2]).should.be.approximately(predictions[2], 3e-1);
        (results[3]).should.be.approximately(predictions[3], 3e-1);
    });

    it('multiclass clasification', function () {
        var trainingSet = [[0, 0], [0, 1], [1, 0], [1, 1]];
        var predictions = [[2], [0], [1], [0]];

        var nn = new FeedforwardNeuralNetwork();
        var options = {
            hiddenLayers: [4],
            iterations: 300,
            learningRate : 0.5,
            momentum: 0.1
        };
        nn.train(trainingSet, predictions, options);

        var result = nn.predict(trainingSet);

        result[0].should.be.approximately(2, 1e-1);
        result[1].should.be.approximately(0, 1e-1);
        result[2].should.be.approximately(1, 1e-1);
        result[3].should.be.approximately(0, 1e-1);
    });

    it('big case', function () {
        var trainingSet = [[1, 1], [1, 2], [2, 1], [2, 2], [3, 1], [1, 3], [1, 4], [4, 1],
                            [6, 1], [6, 2], [6, 3], [6, 4], [6, 5], [5, 5], [4, 5], [3, 5]];
        var predictions = [[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0],
                            [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]];

        var nn = new FeedforwardNeuralNetwork();
        var options = {
            hiddenLayers: [10],
            iterations: 200,
            learningRate : 0.1,
            momentum: 0.1
        };
        nn.train(trainingSet, predictions, options);

        var result = nn.predict([[5, 4]]);

        result[0][0].should.be.lessThan(result[0][1]);
    });

    it('cancer case', function (done) {
        var fs = require('fs');
        var Papa = require('babyparse');
        var data = fs.readFileSync(__dirname + '/matrix.txt', 'utf8');
        data = Papa.parse(data, {
            dynamicTyping: true,
            skipEmptyLines: true,
            header:false
        }).data;

        var dots = new Array(data.length);
        var labels = new Array(data.length);
        for (var i = 0; i < data.length; i++) {
            labels[i] = new Array(2);
            if (data[i][1] === 'M') {
                labels[i] = [1];
            } else {
                labels[i] = [0];
            }
            dots[i] = data[i].slice(2, data[i].length);
        }

        var nn = new FeedforwardNeuralNetwork();
        nn.train(dots, labels, {
            hiddenLayers: [15],
            learningRate: 0.0001,
            iterations: 100,
            momentum: 0.8
        });
        var predictions = nn.predict(dots);

        var error = 0;
        for (var i = 0; i < data.length; i++) {
            predictions[i] = predictions[i] > 0.50 ? 1 : 0;
            if (labels[i][0] !== predictions[i]) {
                error += 1;
            }
        }
        error = error * 100 / labels.length;
        error.should.be.lessThan(20);
        done();
    })
});