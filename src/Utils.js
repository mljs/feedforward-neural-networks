'use strict';

let Matrix = require('ml-matrix');


class Utils {
    /**
     * Apply matrix sum between two matrices and retrieves the result over the first argument matrix.
     * @param {Matrix} A
     * @param {Matrix} B
     * @return {Matrix} A
     */
    static matrixSum(A, B) {
        if (A.rows !== B.rows || A.columns !== B.columns) {
            throw new RangeError('Rows and cols must be the same');
        }

        for (let i = 0; i < A.rows; ++i) {
            for (let j = 0; j < A.columns; ++j) {
                A[i][j] += B[i][j];
            }
        }
        return A;
    }

    /**
     * Apply matrix multiplication element by element and retrieves the result over the first argument matrix.
     * @param {Matrix} A
     * @param {Matrix} B
     * @return {Matrix} A
     */
    static elementWiseMul(A, B) {
        if (A.rows !== B.rows || A.columns !== B.columns) {
            throw new RangeError('Rows and cols must be the same');
        }

        for (let i = 0; i < A.rows; ++i) {
            for (let j = 0; j < A.columns; ++j) {
                A[i][j] *= B[i][j];
            }
        }

        return A;
    }

    /**
     * Retrieves the sum at each row of the given matrix.
     * @param {Matrix} matrix
     * @return {Matrix}
     */
    static sumRow(matrix) {
        let sum = Matrix.zeros(matrix.rows, 1);
        for (let i = 0; i < matrix.rows; ++i) {
            for (let j = 0; j < matrix.columns; ++j) {
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
    static sumCol(matrix) {
        let sum = Matrix.zeros(1, matrix.columns);
        for (let i = 0; i < matrix.rows; ++i) {
            for (let j = 0; j < matrix.columns; ++j) {
                sum[0][j] += matrix[i][j];
            }
        }
        return sum;
    }

    /**
     * Apply scalar multiplication over a matrix (inplace).
     * @param {Matrix} matrix
     * @param {number} scalar
     * @return {Matrix}
     */
    static scalarMul(matrix, scalar) {
        for (let i = 0; i < matrix.rows; ++i) {
            for (let j = 0; j < matrix.columns; ++j) {
                matrix[i][j] *= scalar;
            }
        }
        return matrix;
    }

    /**
     * Method that given an array of labels(predictions), returns two dictionaries, one to transform from labels to
     * numbers and other in the reverse way
     * @param {Array} array
     * @return {object}
     */
    static dictOutputs(array) {
        let inputs = {}, outputs = {}, l = array.length, index = 0;
        for (let i = 0; i < l; i += 1) {
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
}

module.exports = Utils;
