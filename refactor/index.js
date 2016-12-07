"use strict";

let Matrix = require("ml-matrix");

// TODO: the last layer must have an exponential output
class FeedForwardNeuralNetworks {
    constructor(options) {
        if (options === undefined) options = {};

        this.hiddenLayers = options.hiddenLayers === undefined ? [10] : options.hiddenLayers;
        this.iterations = options.iterations === undefined ? 50 : options.iterations;

        this.learningRate = options.learningRate === undefined ? 0.01 : options.learningRate;
        this.momentum = options.momentum === undefined ? 0.1 : options.momentum;
        this.regularization = options.regularization === undefined ? 0.01 : options.regularization;

        this.activationFunction = options.activationFunction === undefined ? Math.tanh : options.activationFunction;
        this.derivateFunction = options.derivateFunction === undefined ? val => 1 - (val * val) : options.derivateFunction;
    }

    buildNetwork(inputSize, outputSize) {
        let size = 2 + (this.hiddenLayers.length - 1);
        this.model = new Array(size);

        // input layer
        this.model[0] = new Layer({
            inputSize: inputSize,
            outputSize: this.hiddenLayers[0],
            activationFunction: this.activationFunction,
            derivate: this.derivateFunction,
            regularization: this.regularization,
            epsilon: this.learningRate
        });

        // hidden layers
        for(let i = 1; i < this.hiddenLayers.length; ++i) {
            this.model[i] = new Layer({
                inputSize: this.hiddenLayers[i - 1],
                outputSize: this.hiddenLayers[i],
                activationFunction: this.activationFunction,
                derivate: this.derivateFunction,
                regularization: this.regularization,
                epsilon: this.learningRate
            });
        }

        // output layer
        this.model[size - 1] = new Layer({
            inputSize: this.hiddenLayers[this.hiddenLayers.length - 1],
            outputSize: outputSize,
            activationFunction: Math.exp,
            derivate: val => val,
            regularization: this.regularization,
            epsilon: this.learningRate
        });
    }

    train(features, labels) {
        features = Matrix.checkMatrix(features);
        this.dicts = Utils.dictOutputs(labels);

        let inputSize = features.columns;
        let outputSize = Object.keys(this.dicts.inputs).length;

        this.buildNetwork(inputSize, outputSize);
        this.model[0].W = new Matrix([[1.24737338, 0.28295388, 0.69207227], [1.58455078, 1.32056292, -0.69103982]]);
        this.model[1].W = new Matrix([[0.5485338, -0.08738612], [-0.05959343,  0.23705916], [0.08316359, 0.8396252]]);
        for(let i = 0; i < this.iterations; ++i) {
            let output = this.propagate(features);
        }
    }

    propagate(X) {
        let input = X;
        for(let j = 0; j < this.model.length; ++j) {
            input = this.model[j].forward(input);
        }

        // get probabilities
        return input.divColumnVector(Utils.sumRow(input))
    }

    predict(features) {

    }

    score() {

    }

    toJSON() {

    }

    static load() {

    }
}

class Layer {
    constructor(options) {
        this.inputSize = options.inputSize;
        this.outputSize = options.outputSize;
        this.regularization = options.regularization;
        this.epsilon = options.epsilon;

        this.activationFunction = function(i, j) {
            this[i][j] = options.activationFunction(this[i][j]);
        };
        this.derivate = function(i, j) {
            this[i][j] = options.derivate(this[i][j]);
        };

        this.W = Matrix.rand(this.inputSize, this.outputSize);
        this.b = Matrix.zeros(1, this.outputSize);

        this.W.apply(function(i, j) {
            this[i][j] /= Math.sqrt(options.inputSize);
        });
    }

    forward(X) {
        let z = X.mmul(this.W).addRowVector(this.b);
        z.apply(this.activationFunction);
        this.a = z.clone();
        return z;
    }

    backpropagation(delta) {
        this.dW = this.a.transpose().mmul(delta);
        this.db = Utils.sumCol(delta);

        let aCopy = this.a.clone();
        let newDelta = delta.mmul(this.W.transpose()).mmul(aCopy.apply(this.derivate));
        return newDelta;
    }

    update() {
        Utils.matrixSum(this.dW, Utils.scalarMul(this.W, this.regularization));
        Utils.matrixSum(this.W, Utils.scalarMul(this.dW, -this.epsilon));
    }
}

class Utils {
    static matrixSum(A, B) {
        if(A.rows !== B.rows || A.cols !== B.cols) {
            throw new RangeError("Rows and cols must be the same");
        }

        for(let i = 0; i < A.rows; ++i) {
            for(let j = 0; j < B.rows; ++j) {
                A[i][j] += B[i][j];
            }
        }
    }

    static sumRow(matrix) {
        let sum = Matrix.zeros(matrix.rows, 1);
        for(let i = 0; i < matrix.rows; ++i) {
            for(let j = 0; j < matrix.columns; ++j) {
                sum[i][0] += matrix[i][j];
            }
        }
        return sum;
    }

    static sumCol(matrix) {
        let sum = Matrix.zeros(1, matrix.cols);
        for(let i = 0; i < matrix.rows; ++i) {
            for(let j = 0; j < matrix.columns; ++j) {
                sum[0][j] += matrix[i][j];
            }
        }
        return sum;
    }

    static scalarMul(matrix, scalar) {
        for(let i = 0; i < matrix.rows; ++i) {
            for(let j = 0; j < matrix.cols; ++j) {
                matrix[i][j] *= scalar;
            }
        }
    }

    static dictOutputs(array) {
        let inputs = {} , outputs = {}, l = array.length, index = 0;
        for(let i = 0; i < l; i += 1) {
            if(inputs[array[i]] === undefined) {
                inputs[array[i]] = index;
                outputs[index] = array[i];
                index++;
            }
        }

        return {
            inputs: inputs,
            outputs: outputs
        };
    }
}

let X = [[0, 0], [0, 1], [1, 0], [1, 1]];
let y = [0, 0, 0, 1];

let fnn = new FeedForwardNeuralNetworks({
    hiddenLayers: [3],
    iterations: 20000
});

fnn.train(X, y);