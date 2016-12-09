'use strict';

function logistic(val) {
    return 1 / (1 + Math.exp(-val));
}

const ACTIVATION_FUNCTIONS = {
    'tanh': {
        activation: Math.tanh,
        derivate: val => 1 - (val * val)
    },
    'identity': {
        activation: val => val,
        derivate: () => 1
    },
    'logistic': {
        activation: logistic,
        derivate: val => logistic(val) * (1 - logistic(val))
    },
    'arctan': {
        activation: Math.atan,
        derivate: val => 1 / (val * val + 1)
    },
    'softsign': {
        activation: val => val / (1 + Math.abs(val)),
        derivate: val => 1 / ((1 + Math.abs(val)) * (1 + Math.abs(val)))
    },
    'relu': {
        activation: val => val < 0 ? 0 : val,
        derivate: val => val < 0 ? 0 : 1
    },
    'softplus': {
        activation: val => Math.log(1 + Math.exp(val)),
        derivate: val => 1 / (1 + Math.exp(-val))
    },
    'bent': {
        activation: val => ((Math.sqrt(val * val + 1) - 1) / 2) + val,
        derivate: val => (val / (2 * Math.sqrt(val * val + 1))) + 1
    },
    'sinusoid': {
        activation: Math.sin,
        derivate: Math.cos
    },
    'sinc': {
        activation: val => val === 0 ? 1 : Math.sin(val) / val,
        derivate: val => val === 0 ? 0 : (Math.cos(val) / val) - (Math.sin(val) / (val * val))
    },
    'gaussian': {
        activation: val => Math.exp(-(val * val)),
        derivate: val => -2 * val * Math.exp(-(val * val))
    }
};

module.exports = ACTIVATION_FUNCTIONS;
