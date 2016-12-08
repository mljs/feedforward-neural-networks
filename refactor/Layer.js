"use strict";

let Matrix = require("ml-matrix");

let Utils = require("./Utils");
const ACTIVATION_FUNCTIONS = require("./ActivationFunctions");

class Layer {
    constructor(options) {
        this.inputSize = options.inputSize;
        this.outputSize = options.outputSize;
        this.regularization = options.regularization;
        this.epsilon = options.epsilon;
        this.activation = options.activation;

        this.activationFunction = function(i, j) {
            this[i][j] = ACTIVATION_FUNCTIONS[options.activation].activation(this[i][j]);
        };
        this.derivate = function(i, j) {
            this[i][j] = ACTIVATION_FUNCTIONS[options.activation].derivate(this[i][j]);
        };

        if(options.model) {
            // load model
            this.W = Matrix.checkMatrix(options.W);
            this.b = Matrix.checkMatrix(options.b);

        } else {
            // default constructor

            this.W = Matrix.rand(this.inputSize, this.outputSize);
            this.b = Matrix.zeros(1, this.outputSize);

            this.W.apply(function(i, j) {
                this[i][j] /= Math.sqrt(options.inputSize);
            });
        }
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
        Utils.matrixSum(this.dW, Utils.scalarMul(this.W.clone(), this.regularization));
        Utils.matrixSum(this.W, Utils.scalarMul(this.dW, -this.epsilon));
        Utils.matrixSum(this.b, Utils.scalarMul(this.db, -this.epsilon));
    }

    toJSON() {
        return {
            model: "Layer",
            inputSize: this.inputSize,
            outputSize: this.outputSize,
            regularization: this.regularization,
            epsilon: this.epsilon,
            activation: this.activation,
            W: this.W,
            b: this.b
        };
    }

    static load(model) {
        if(model.model !== "Layer") {
            throw new RangeError("the current model is not a Layer model");
        }
        return new Layer(model);
    }

}

module.exports = Layer;