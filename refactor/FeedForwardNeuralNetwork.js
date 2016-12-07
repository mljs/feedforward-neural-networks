"use strict";

let Matrix = require("ml-matrix");
let Layer = require("./Layer");
let Utils = require("./Utils");

class FeedForwardNeuralNetworks {
    constructor(options) {
        if (options === undefined) options = {};

        this.hiddenLayers = options.hiddenLayers === undefined ? [10] : options.hiddenLayers;
        this.iterations = options.iterations === undefined ? 50 : options.iterations;

        this.learningRate = options.learningRate === undefined ? 0.01 : options.learningRate;
        //this.momentum = options.momentum === undefined ? 0.1 : options.momentum;
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
        //this.model[0].W = new Matrix([[1.24737338, 0.28295388, 0.69207227], [1.58455078, 1.32056292, -0.69103982]]);
        //this.model[1].W = new Matrix([[0.5485338, -0.08738612], [-0.05959343,  0.23705916], [0.08316359, 0.8396252]]);
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

        for(let i = 0; i < this.model.length; ++i) {
            this.model[i].update();
        }
    }

    predict(features) {
        Matrix.checkMatrix(features);
        let outputs = new Array(features.rows);
        let probabilities = this.propagate(features);
        for(let i = 0; i < features.rows; ++i) {
            outputs[i] = this.dicts.outputs[probabilities.maxRowIndex(i)[1]];
        }

        return outputs;
    }

    score() {

    }

    toJSON() {

    }

    static load() {

    }
}

module.exports = FeedForwardNeuralNetworks;