
import { Matrix } from 'ml-matrix';

import ACTIVATION_FUNCTIONS from '../activationFunctions';
import FeedforwardNeuralNetwork from '../FeedForwardNeuralNetwork';

describe('Feedforward Neural Networks', function () {
  var functions = Object.keys(ACTIVATION_FUNCTIONS);

  it('Training the neural network with XOR operator', function () {
    var trainingSet = new Matrix([[0, 0], [0, 1], [1, 0], [1, 1]]);
    var predictions = [false, true, true, false];

    for (var i = 0; i < functions.length; ++i) {
      var options = {
        hiddenLayers: [4],
        iterations: 500,
        learningRate: 0.3,
        activation: functions[i]
      };
      var xorNN = new FeedforwardNeuralNetwork(options);

      xorNN.train(trainingSet, predictions);
      var results = xorNN.predict(trainingSet);

      expect(results[0]).toBe(predictions[0]);
      expect(results[1]).toBe(predictions[1]);
      expect(results[2]).toBe(predictions[2]);
      expect(results[3]).toBe(predictions[3]);
    }
  });

  it('Training the neural network with AND operator', function () {
    var trainingSet = [[0, 0], [0, 1], [1, 0], [1, 1]];
    var predictions = [[1, 0], [1, 0], [1, 0], [0, 1]];

    for (var i = 0; i < functions.length; ++i) {
      var options = {
        hiddenLayers: [3],
        iterations: 500,
        learningRate: 0.3,
        activation: functions[i]
      };
      var andNN = new FeedforwardNeuralNetwork(options);
      andNN.train(trainingSet, predictions);

      var results = andNN.predict(trainingSet);

      expect(results[0][0]).toBeGreaterThan(results[0][1]);
      expect(results[1][0]).toBeGreaterThan(results[1][1]);
      expect(results[2][0]).toBeGreaterThan(results[2][1]);
      expect(results[3][0]).toBeLessThan(results[3][1]);
    }
  });

  it('Export and import', function () {
    var trainingSet = [[0, 0], [0, 1], [1, 0], [1, 1]];
    var predictions = [0, 1, 1, 1];

    for (var i = 0; i < functions.length; ++i) {
      var options = {
        hiddenLayers: [4],
        iterations: 500,
        learningRate: 0.3,
        activation: functions[i]
      };
      var orNN = new FeedforwardNeuralNetwork(options);
      orNN.train(trainingSet, predictions);

      var model = JSON.parse(JSON.stringify(orNN));
      var networkNN = FeedforwardNeuralNetwork.load(model);

      var results = networkNN.predict(trainingSet);

      expect(results[0]).toBeCloseTo(predictions[0], 2);
      expect(results[1]).toBeCloseTo(predictions[1], 2);
      expect(results[2]).toBeCloseTo(predictions[2], 2);
      expect(results[3]).toBeCloseTo(predictions[3], 2);
    }
  });

  it('Multiclass clasification', function () {
    var trainingSet = [[0, 0], [0, 1], [1, 0], [1, 1]];
    var predictions = [2, 0, 1, 0];

    for (var i = 0; i < functions.length; ++i) {
      var options = {
        hiddenLayers: [4],
        iterations: 300,
        learningRate: 0.5,
        activation: functions[i]
      };
      var nn = new FeedforwardNeuralNetwork(options);
      nn.train(trainingSet, predictions);

      var result = nn.predict(trainingSet);

      expect(result[0]).toBeCloseTo(2, 1);
      expect(result[1]).toBeCloseTo(0, 1);
      expect(result[2]).toBeCloseTo(1, 1);
      expect(result[3]).toBeCloseTo(0, 1);
    }
  });

  it('Big case', function () {
    var trainingSet = [
      [1, 1], [1, 2], [2, 1], [2, 2], [3, 1], [1, 3], [1, 4], [4, 1],
      [6, 1], [6, 2], [6, 3], [6, 4], [6, 5], [5, 5], [4, 5], [3, 5]
    ];
    var predictions = [
      [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0],
      [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]
    ];
    for (var i = 0; i < functions.length; ++i) {
      var options = {
        hiddenLayers: [20],
        iterations: 1000,
        learningRate: 0.01,
        activation: functions[i]
      };
      var nn = new FeedforwardNeuralNetwork(options);
      nn.train(trainingSet, predictions);

      var result = nn.predict([[5, 4]]);

      expect(result[0][0]).toBeLessThan(result[0][1]);
    }
  });

  it('Big case - many predictions', function () {
    var trainingSet = [
      [1, 1], [1, 2], [2, 1], [2, 2], [3, 1], [1, 3], [1, 4], [4, 1],
      [6, 1], [6, 2], [6, 3], [6, 4], [6, 5], [5, 5], [4, 5], [3, 5]
    ];
    var predictions = [
      [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0],
      [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]
    ];
    for (var i = 0; i < functions.length; ++i) {
      var options = {
        hiddenLayers: [20],
        iterations: 1000,
        learningRate: 0.01,
        activation: functions[i]
      };
      var nn = new FeedforwardNeuralNetwork(options);
      nn.train(trainingSet, predictions);

      var result = nn.predict([[5, 4], [4, 2], [5, 3]]);
      expect(result[0][0]).toBeLessThan(result[0][1]);
      expect(result[1][1]).toBeLessThan(result[1][0]);
      expect(result[2][0]).toBeLessThan(result[2][1]);
    }
  });

  it.skip('Big test case 2', function () {
    // see https://gist.github.com/jajoe/cb53d7b1378a76cc6896e660f83b50d2, this test case should work
    var X = [[0, 255, 255, 255, 0], [255, 0, 0, 0, 255], [255, 255, 0, 0, 0], [255, 0, 0, 0, 0], [0, 255, 255, 255, 255], [0, 255, 255, 0, 0], [0, 255, 0, 0, 255], [0, 0, 255, 0, 255], [255, 255, 0, 0, 255], [255, 0, 0, 0, 255], [255, 255, 0, 255, 0], [0, 0, 0, 255, 0], [255, 0, 0, 255, 0], [255, 0, 0, 255, 255], [0, 0, 255, 0, 255]];
    var y = [[0, 1], [1, 0], [1, 1], [1, 1], [1, 1], [0, 1], [0, 0], [0, 0], [1, 0], [1, 0], [0, 0], [0, 1], [0, 0], [1, 0], [0, 0]];
    var Xtest = [[255, 0, 255, 255, 255], [0, 255, 0, 0, 0], [255, 0, 255, 255, 0], [0, 0, 255, 255, 255]];
    var ytest = [[1, 0], [0, 1], [0, 0], [1, 1]];
    var options = {
      hiddenLayers: [100],
      iterations: 10000,
      learningRate: 0.001,
      activation: 'logistic'
    };
    var nn = new FeedforwardNeuralNetwork(options);
    nn.train(X, y);

    var result = nn.predict(Xtest);
    for (let i = 0; i < ytest.length; i++) {
      for (let j = 0; j < ytest[0].length; j++) {
        expect(result[i][j]).toBeCloseTo(ytest[i][j], 1);
      }
    }
  });
});
