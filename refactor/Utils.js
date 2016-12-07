"use strict";

let Matrix = require("ml-matrix");

class Utils {
    static matrixSum(A, B) {
        if(A.rows !== B.rows || A.columns !== B.columns) {
            throw new RangeError("Rows and cols must be the same");
        }

        for(let i = 0; i < A.rows; ++i) {
            for(let j = 0; j < A.columns; ++j) {
                A[i][j] += B[i][j];
            }
        }
        return A;
    }

    static elementWiseMul(A, B) {
        if(A.rows !== B.rows || A.columns !== B.columns) {
            throw new RangeError("Rows and cols must be the same");
        }

        for(let i = 0; i < A.rows; ++i) {
            for(let j = 0; j < A.columns; ++j) {
                A[i][j] *= B[i][j];
            }
        }

        return A;
    }

    static sumRow(matrix) {
        let sum = Matrix.zeros(matrix.rows, 1);
        for(let i = 0; i < matrix.rows; ++i) {
            for(let j = 0; j < matrix.columns; ++j) {
                sum[i][0] += matrix[i][j];
            }
        }
        return sum;
    }

    static sumCol(matrix) {
        let sum = Matrix.zeros(1, matrix.columns);
        for(let i = 0; i < matrix.rows; ++i) {
            for(let j = 0; j < matrix.columns; ++j) {
                sum[0][j] += matrix[i][j];
            }
        }
        return sum;
    }

    static scalarMul(matrix, scalar) {
        for(let i = 0; i < matrix.rows; ++i) {
            for(let j = 0; j < matrix.columns; ++j) {
                matrix[i][j] *= scalar;
            }
        }
        return matrix;
    }

    static dictOutputs(array) {
        let inputs = {} , outputs = {}, l = array.length, index = 0;
        for(let i = 0; i < l; i += 1) {
            if(inputs[array[i]] === undefined) {
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