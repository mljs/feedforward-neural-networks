"use strict";

let Matrix = require("ml-matrix");

let Layer = require("./Layer");
let Utils = require("./Utils");
const ACTIVATION_FUNCTIONS = require("./ActivationFunctions");

class FeedForwardNeuralNetworks {
    constructor(options) {
        if (options === undefined) options = {};
        if(options.model) {
            // load network
            this.hiddenLayers = options.hiddenLayers;
            this.iterations = options.iterations;
            this.learningRate = options.learningRate;
            this.regularization = options.regularization;
            this.dicts = options.dicts;
            this.activation = options.activation;
            this.model = new Array(options.layers.length);

            for(let i = 0; i < this.model.length; ++i) {
                this.model[i] = Layer.load(options.layers[i]);
            }
        } else {
            // default constructor
            this.hiddenLayers = options.hiddenLayers === undefined ? [10] : options.hiddenLayers;
            this.iterations = options.iterations === undefined ? 50 : options.iterations;

            this.learningRate = options.learningRate === undefined ? 0.01 : options.learningRate;
            //this.momentum = options.momentum === undefined ? 0.1 : options.momentum;
            this.regularization = options.regularization === undefined ? 0.01 : options.regularization;

            this.activation = options.activation === undefined ? "tanh" : options.activation;
            if(!this.activation in Object.keys(ACTIVATION_FUNCTIONS)) {
                console.warn("Setting default activation function: 'tanh'");
                this.activation = "tanh";
            }
        }
    }

    buildNetwork(inputSize, outputSize) {
        let size = 2 + (this.hiddenLayers.length - 1);
        this.model = new Array(size);

        // input layer
        this.model[0] = new Layer({
            inputSize: inputSize,
            outputSize: this.hiddenLayers[0],
            activation: this.activation,
            regularization: this.regularization,
            epsilon: this.learningRate
        });

        // hidden layers
        for(let i = 1; i < this.hiddenLayers.length; ++i) {
            this.model[i] = new Layer({
                inputSize: this.hiddenLayers[i - 1],
                outputSize: this.hiddenLayers[i],
                activation: this.activation,
                regularization: this.regularization,
                epsilon: this.learningRate
            });
        }

        // output layer
        this.model[size - 1] = new Layer({
            inputSize: this.hiddenLayers[this.hiddenLayers.length - 1],
            outputSize: outputSize,
            activation: "exp",
            regularization: this.regularization,
            epsilon: this.learningRate
        });
    }

    train(X, y) {
        X = Matrix.checkMatrix(X);
        this.dicts = Utils.dictOutputs(y);

        let inputSize = X.columns;
        let outputSize = Object.keys(this.dicts.inputs).length;

        this.buildNetwork(inputSize, outputSize);

        for(let i = 0; i < this.iterations; ++i) {
            let probabilities = this.propagate(X);
            this.backpropagation(X, y, probabilities);
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

    predict(X) {
        X = Matrix.checkMatrix(X);
        let outputs = new Array(X.rows);
        let probabilities = this.propagate(X);
        for(let i = 0; i < X.rows; ++i) {
            outputs[i] = this.dicts.outputs[probabilities.maxRowIndex(i)[1]];
        }

        return outputs;
    }

    score() {

    }

    toJSON() {
        let model = {
            model: "FNN",
            hiddenLayers: this.hiddenLayers,
            iterations: this.iterations,
            learningRate: this.learningRate,
            regularization: this.regularization,
            activation: this.activation,
            dicts: this.dicts,
            layers: new Array(this.model.length)
        };

        for(let i = 0; i < this.model.length; ++i) {
            model.layers[i] = this.model[i].toJSON();
        }

        return model;
    }


    static load(model) {
        if(model.model !== "FNN") {
            throw new RangeError("the current model is not a feed forward network");
        }

        return new FeedForwardNeuralNetworks(model);
    }
}

module.exports = FeedForwardNeuralNetworks;