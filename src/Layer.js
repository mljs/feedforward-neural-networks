'use strict';

var Matrix = require('ml-matrix');

var Utils = require('./utils');
const ACTIVATION_FUNCTIONS = require('./activationFunctions');

class Layer {
    /**
     * Create a new layer with the given options
     * @param {object} options
     * @param {number} [options.inputSize] - Number of conections that enter the neurons.
     * @param {number} [options.outputSize] - Number of conections that leave the neurons.
     * @param {number} [options.regularization] - Regularization parameter.
     * @param {number} [options.epsilon] - Learning rate parameter.
     * @param {string} [options.activation] - Activation function parameter from the FeedForwardNeuralNetwork class.
     * @param {number} [options.activationParam] - Activation parameter if needed.
     */
    constructor(options) {
        this.inputSize = options.inputSize;
        this.outputSize = options.outputSize;
        this.regularization = options.regularization;
        this.epsilon = options.epsilon;
        this.activation = options.activation;
        this.activationParam = options.activationParam;

        var selectedFunction = ACTIVATION_FUNCTIONS[options.activation];
        var params = selectedFunction.activation.length;

        var actFunction = params > 1 ? val => selectedFunction.activation(val, options.activationParam) : selectedFunction.activation;
        var derFunction = params > 1 ? val => selectedFunction.derivate(val, options.activationParam) : selectedFunction.derivate;

        this.activationFunction = function (i, j) {
            this[i][j] = actFunction(this[i][j]);
        };
        this.derivate = function (i, j) {
            this[i][j] = derFunction(this[i][j]);
        };

        if (options.model) {
            // load model
            this.W = Matrix.checkMatrix(options.W);
            this.b = Matrix.checkMatrix(options.b);

        } else {
            // default constructor

            this.W = Matrix.rand(this.inputSize, this.outputSize);
            this.b = Matrix.zeros(1, this.outputSize);

            this.W.apply(function (i, j) {
                this[i][j] /= Math.sqrt(options.inputSize);
            });
        }
    }

    /**
     * propagate the given input through the current layer.
     * @param {Matrix} X - input.
     * @return {Matrix} output at the current layer.
     */
    forward(X) {
        var z = X.mmul(this.W).addRowVector(this.b);
        z.apply(this.activationFunction);
        this.a = z.clone();
        return z;
    }

    /**
     * apply backpropagation algorithm at the current layer
     * @param {Matrix} delta - delta values estimated at the following layer.
     * @param {Matrix} a - 'a' values from the following layer.
     * @return {Matrix} the new delta values for the next layer.
     */
    backpropagation(delta, a) {
        this.dW = a.transposeView().mmul(delta);
        this.db = Utils.sumCol(delta);

        var aCopy = a.clone();
        return delta.mmul(this.W.transposeView()).mul(aCopy.apply(this.derivate));
    }

    /**
     * Function that updates the weights at the current layer with the derivatives.
     */
    update() {
        this.dW.add(this.W.clone().mul(this.regularization));
        this.W.add(this.dW.mul(-this.epsilon));
        this.b.add(this.db.mul(-this.epsilon));
    }

    /**
     * Export the current layer to JSON.
     * @return {object} model
     */
    toJSON() {
        return {
            model: 'Layer',
            inputSize: this.inputSize,
            outputSize: this.outputSize,
            regularization: this.regularization,
            epsilon: this.epsilon,
            activation: this.activation,
            W: this.W,
            b: this.b
        };
    }

    /**
     * Creates a new Layer with the given model.
     * @param {object} model
     * @return {Layer}
     */
    static load(model) {
        if (model.model !== 'Layer') {
            throw new RangeError('the current model is not a Layer model');
        }
        return new Layer(model);
    }

}

module.exports = Layer;
