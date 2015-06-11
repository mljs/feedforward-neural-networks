//TODO: use strict
var Matrix = require('ml-matrix');

FeedforwardNeuralNetworks = function() {
    var X;
    var y;
    var Theta = new Array(2);
    var lambda;

    var Sigmoid = function Sigmoid(value) {
        return 1.0 / (1 + Math.exp(-value));
    };

    var SigmoidGradient = function SigmoidGradient(value) {
        var sig = Sigmoid(value);
        return sig * (1 - sig);
    };

    var sigmoid = function (i, j) {
        this[i][j] = Sigmoid(this[i][j]);
        return this;
    };

    var sigmoidGradient = function(i, j) {
        this[i][j] = SigmoidGradient(this[i][j]);
        return this;
    };

    var logArray = function(i, j) {
        this[i][j] = Math.log(this[i][j]);
        return this;
    };

    var costFunction = function costFunction(X, y, lambdaArg, numberOfLabels) {
        var m = X.rows;
        var grad = new Array(2);
        grad[0] = Matrix.zeros(Theta[0].rows, Theta[0].columns);
        grad[1] = Matrix.zeros(Theta[1].rows, Theta[1].columns);

        X.addColumn(Matrix.ones(m, 1));
        var transposeX = X.transpose();

        var a2 = Theta[0].mmul(transposeX).apply(sigmoid);
        a2 = a2.transpose().addColumn(Matrix.ones(m, 1));

        var a3 = Theta[1].mmul(a2.transpose()).apply(sigmoid);
        a3 = a3.transpose();

        var yk = Matrix.zeros(numberOfLabels, m);
        for(var i = 0; i < m; ++i) {
            yk[y[i]][i] = 1;
        }
        var negativeYk = yk.clone().neg();

        // TODO: review the cost calculation
        /*var positiveCost = negativeYk.mulM(a3.apply(logArray));
        var negativeCost =*/
        var cost = 0;

        for(i = 0; i < m; ++i) {
            var a1 = Matrix.rowVector(X.getRow(i));

            var z2 = Theta[0].mmul(a1.transpose());
            a2 = z2.clone().apply(sigmoid);

            a2 = a2.transpose().addColumn(0, Matrix.ones(1, 1));

            var z3 = Theta[1].mmul(a2.transpose());
            a3 = z3.clone().apply(sigmoid);

            var delta3 = a3.clone().add(Matrix.columnVector(negativeYk.getColumn(i)));

            z2.addRow(0, Matrix.ones(1, 1));

            var delta2 = Theta[1].transpose().mmul(delta3).mulM(z2.clone().apply(sigmoidGradient));
            delta2.removeRow(0); // don't forget it

            grad[0].add((delta2.mmul(a1)));
            grad[1].add((delta3.mmul(a2)));
        }

        grad[0].addS(1 / m);
        grad[1].addS(1 / m);

        X.removeColumn(0); // Removing the bias term from the training set

        return {cost: cost, grad: grad};
    };

    function randomInitialzeTheta(labelsIn, labelsOut) {
        var epsilonRange = 0.12; // values around

        return Matrix.rand(labelsOut, labelsIn).mulS(2).mulS(epsilonRange).addS(-epsilonRange);
    }

    this.train = function(XArg, yArg, learningRate, lambdaArg, numberOfLabels, iterations, hiddenLayerSize) {
        X = XArg;
        y = yArg;
        var m = XArg.rows; // test cases
        var features = XArg.columns;

        Theta[0] = randomInitialzeTheta(features + 1, hiddenLayerSize); // TODO: be careful
        Theta[1] = randomInitialzeTheta(hiddenLayerSize + 1, numberOfLabels);

        var result;
        var previousCost = Infinity;
        var minError = 1e-5;

        for(var i = 0; i < iterations; ++i) {
            var temporalResult = costFunction(X, y, lambdaArg, numberOfLabels);

            /*if(previousCost - temporalResult.cost < minError || previousCost < temporalResult.cost)
                break;
            else {*/
                result = temporalResult;
                previousCost = result.cost;
            /*}*/

            //console.log("cost: " + result.cost);
            Theta[0].add(result.grad[0].mulS(-learningRate));
            Theta[1].add(result.grad[1].mulS(-learningRate));
        }
    };

    this.predict = function(X) {
        var m = X.rows;
        var predictions = Matrix.zeros(m, 1);

        var h1 = X.addColumn(0, Matrix.ones(m, 1)).mmul(Theta[0].transpose()).apply(sigmoid);
        var h2 = h1.addColumn(0, Matrix.ones(m, 1)).mmul(Theta[1].transpose()).apply(sigmoid);

        return h2;
    };

    this.numericalGradient = function(costFunction, theta) {

    };

    return this;
};
