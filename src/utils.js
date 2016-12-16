'use strict';

var Matrix = require('ml-matrix');

/**
 * Retrieves the sum at each row of the given matrix.
 * @param {Matrix} matrix
 * @return {Matrix}
 */
function sumRow(matrix) {
    var sum = Matrix.zeros(matrix.rows, 1);
    for (var i = 0; i < matrix.rows; ++i) {
        for (var j = 0; j < matrix.columns; ++j) {
            sum[i][0] += matrix[i][j];
        }
    }
    return sum;
}

/**
 * Retrieves the sum at each column of the given matrix.
 * @param {Matrix} matrix
 * @return {Matrix}
 */
function sumCol(matrix) {
    var sum = Matrix.zeros(1, matrix.columns);
    for (var i = 0; i < matrix.rows; ++i) {
        for (var j = 0; j < matrix.columns; ++j) {
            sum[0][j] += matrix[i][j];
        }
    }
    return sum;
}

/**
 * Method that given an array of labels(predictions), returns two dictionaries, one to transform from labels to
 * numbers and other in the reverse way
 * @param {Array} array
 * @return {object}
 */
function dictOutputs(array) {
    var inputs = {}, outputs = {}, l = array.length, index = 0;
    for (var i = 0; i < l; i += 1) {
        if (inputs[array[i]] === undefined) {
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

module.exports = {
    dictOutputs: dictOutputs,
    sumCol: sumCol,
    sumRow: sumRow
};
