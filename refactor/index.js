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
            derivate: this.derivateFunction,
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
            let probabilities = this.propagate(features);
            this.backpropagation(features, labels, probabilities);
        }
    }

    propagate(X) {
        let input = X;
        for(let i = 0; i < this.model.length; ++i) {
            input = this.model[i].forward(input);
        }

        // get probabilities
        return input.divColumnVector(Utils.sumRow(input))
    }

    backpropagation(features, labels, probabilities) {
        for(let i = 0; i < probabilities.length; ++i) {
            probabilities[i][this.dicts.inputs[labels[i]]] -= 1;
        }

        // remember, the last delta doesn't matter
        let delta = probabilities;
        for(let i = this.model.length - 1; i >= 0; --i) {
            let a = i > 0 ? this.model[i - 1].a : features;
            delta = this.model[i].backpropagation(delta, a);
        }


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

    backpropagation(delta, a) {
        this.dW = a.transpose().mmul(delta);
        this.db = Utils.sumCol(delta);

        let aCopy = a.clone();
        return Utils.elementWiseMul(delta.mmul(this.W.transpose()), aCopy.apply(this.derivate));
    }

    update() {
        Utils.matrixSum(this.dW, Utils.scalarMul(this.W, this.regularization));
        Utils.matrixSum(this.W, Utils.scalarMul(this.dW, -this.epsilon));
        Utils.matrixSum(this.b, Utils.scalarMul(this.db, -this.epsilon));
    }
}

class Utils {
    static matrixSum(A, B) {
        if(A.rows !== B.rows || A.columns !== B.columns) {
            throw new RangeError("Rows and cols must be the same");
        }

        for(let i = 0; i < A.rows; ++i) {
            for(let j = 0; j < A.columns; ++j) {
                A[i][j] += B[i][j];
            }
        }
        return A;
    }

    static elementWiseMul(A, B) {
        if(A.rows !== B.rows || A.columns !== B.columns) {
            throw new RangeError("Rows and cols must be the same");
        }

        for(let i = 0; i < A.rows; ++i) {
            for(let j = 0; j < A.columns; ++j) {
                A[i][j] *= B[i][j];
            }
        }

        return A;
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
        let sum = Matrix.zeros(1, matrix.columns);
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