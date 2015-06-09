//TODO: use strict
var Matrix = require('ml-matrix');

FeedforwardNeuralNetworks = function() {

    var X;
    var y;
    var Theta = new Array(2);
    var alpha;
    var lambda;

    function Sigmoid(value) {
        return 1.0 / (1 + Math.exp(value));
    }

    function SigmoidGradient(value) {
        var sig = Sigmoid(value);
        return sig * (1 - sig);
    }

    var sigmoid = function (i, j) {
        this[i][j] = Sigmoid(this[i][j]);
        return this;
    };

    var sigmoidGradient = function(i, j) {
        this[i][j] = Sigmoid(this[i][j]);
        return this;
    };

    var costFunction = function costFunction(X, y, lambdaArg, numberOfLabels) {
        lambda = lambdaArg;

        var m = X.rows;
        var cost;

        X = X.addColumn(0, Matrix.ones(m, 1));
        var transposeX = X.transpose();

        var a2 = (Theta[0].mmul(transposeX)).apply(sigmoid);

        a2 = a2.transpose().addColumn(0, Matrix.ones(m, 1));

        var a3 = Theta[1].mmul(a2.transpose()).apply(sigmoid);
        a3 = a3.transpose();
        var yk = Matrix.zeros(numberOfLabels, m);

        for(var i = 0; i < m; ++i) {
            yk[y[i]][i] = 1;
        }

        var negativeYk = yk.mulS(-1);

        cost = (1 / m) * (negativeYk.transpose().mulM(a3.apply(Math.log)).subM
            ((negativeYk.addS(1).transpose().mulM(a3.mulS(-1).addS(1))))).sum();

        // Apply regularization

        var grad = new Array(2);
        grad[0] = Matrix.zeros(Theta[0].rows, Theta[0].columns);
        grad[1] = Matrix.zeros(Theta[1].rows, Theta[1].columns);

        negativeYk.neg();

        // Doing backpropagation
        for(i = 0; i < m; ++i) {
            var a1 = Matrix.columnVector(transposeX.getColumn(i));
            var z2 = Theta[0].mmul(a1);

            a2 = z2.apply(sigmoid);
            a2 = a2.transpose().addColumn(0, Matrix.ones(1, 1));

            var z3 = Theta[1].mmul(a2.transpose());
            a3 = z3.apply(sigmoid);

            var delta3 = a3.addColumnVector(Matrix.columnVector(negativeYk.getColumn(i)));
            z2 = z2.addRow(0, Matrix.ones(1, 1));


            var delta2 = Theta[1].transpose().mmul(delta3).mulM(
                            z2.apply(sigmoidGradient));
            delta2 = delta2.removeRow(0);

            // TODO: be careful applying the gradient
            grad[0] = grad[0].addM(delta2.mmul(a1.transpose()));
            grad[1] = grad[1].addM(delta3.mmul(a2));
        }

        grad[0] = grad[0].mulS(1 / m);
        grad[1] = grad[1].mulS(1 / m);

        // removing the bias terms in the training set
        X = X.removeColumn(0);

        // TODO: apply regularization term to the gradient
        var gradBiasTerm = new Array(2);

        // save the bias terms because we don't want to apply the regularization to those terms
        gradBiasTerm[0] = grad[0].getColumn(0);
        gradBiasTerm[1] = grad[1].getColumn(0);

        grad[0] = grad[0].removeColumn(0);
        grad[1] = grad[1].removeColumn(0);

        grad[0].add(Theta[0].subMatrix(0, Theta[0].rows - 1, 1, Theta[0].columns - 1).mulS(lambda / m));
        grad[1].add(Theta[1].subMatrix(0, Theta[1].rows - 1, 1, Theta[1].columns - 1).mulS(lambda / m));

        // add again the regularization terms to gradient vector
        grad[0] = grad[0].addColumn(0, gradBiasTerm[0]);
        grad[1] = grad[1].addColumn(0, gradBiasTerm[1]);
        return {cost: cost, grad: grad};
    };

    function randomInitialzeTheta(labelsIn, labelsOut) {
        var epsilonRange = 0.12; // values around

        return Matrix.rand(labelsOut, labelsIn).mulS(2).mulS(epsilonRange).addS(-epsilonRange);
    }

    this.train = function(XArg, yArg, learningRate, lambdaArg, numberOfLabels, iterations, hiddenLayerSize) {
        X = XArg;
        y = yArg;
        var m = XArg.rows;
        var features = XArg.columns;

        Theta[0] = randomInitialzeTheta(features + 1, hiddenLayerSize); // TODO: be careful
        Theta[1] = randomInitialzeTheta(hiddenLayerSize + 1, numberOfLabels);

        var result;

        for(var i = 0; i < iterations; ++i) {
            result = costFunction(X, y, lambdaArg, numberOfLabels);
            Theta[0].add(result.grad[0].mulS(learningRate / m));
            Theta[1].add(result.grad[1].mulS(learningRate / m));
        }
    };

    this.predict = function(X) {
        var m = X.rows;
        var predictions = Matrix.zeros(m, 1);

        var h1 = X.addColumn(0, Matrix.ones(m, 1)).mmul(Theta[0].transpose()).apply(sigmoid);
        var h2 = h1.addColumn(0, Matrix.ones(m, 1)).mmul(Theta[1].transpose()).apply(sigmoid);

        return h2;
    };

    return this;
};
