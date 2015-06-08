"use strict";

require("../src/util");

describe('Utilities', function() {
    it('Sigmoid function', function() {
        var result = Util.sigmoid(3);
        result.should.be.approximately(0.9525, 1e-4);
    });

    it('sigmoid gradient function', function() {
        var result = Util.sigmoidGradient(3);
        result.should.be.approximately(0.04517, 1e-4);
    });
});