/**
 * Created by jefferson on 6/8/15.
 */
"use strict";

require("../src/util");

describe('Utilities', function() {
    it('Sigmoid function', function() {
        var result = Util.sigmoid(3);
        result.should.be.approximately(0.9525, 1e-4);
    });
});