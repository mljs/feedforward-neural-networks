"use strict";

const ACTIVATION_FUNCTIONS = {
    "tanh": {
        activation: Math.tanh,
        derivate: val => 1 - (val * val)
    },
    "exp": {
        activation: Math.exp,
        derivate: val => 1 - (val * val)
    }
};

module.exports = ACTIVATION_FUNCTIONS;