var Util = require('util');

var FeedforwardNeuralNetworks = function() {

    var X;
    var y;
    var Theta = new Array(2);
    var alpha;
    var lambda;

    function costFunction(lambdaArg, numberOfLabels) {
        lambda = lambdaArg;

        var m = X.rows;
        var cost = 0.0;

        var X = X.addColumn(0, Matrix.ones(m, 1));

        var a2 = Theta[0].mmul(X.transpose()).apply(Util.sigmoid);
        a2 = a2.transpose().addColumn(0, Matrix.ones(m, 1));

        var a3 = Theta[1].mmul(a2.transpose()).apply(Util.sigmoid);
        a3 = a3.transpose();

        var yk = Matrix.zeros(numberOfLabels, m);

        for(var i = 0; i < m; ++i) {
            yk[y[i]][i] = 1;
        }

        cost = (1 / m) * (yk.neg().transpose().mulM(a3.apply(Math.log)) -
            (yk.neg().add(1).transpose().mulM(a3.neg().add(1))));

        // Apply regularization

        var grad = new Array(2);
        grad[0] = 0;
        grad[1] = 0;

        for(i = 0; i < m; ++i) {
            a1 = X.getRow(i);
            z2 = Theta[0].mmul(a1.transpose());

            a2 = z2.apply(Util.sigmoid);
            a2 = a2.transpose().addColumn(0, Matrix.ones(1, 1));

            z3 = Theta[1].mmul(a2.transpose());
            a3 = z3.apply(Util.sigmoid);

            var delta3 = a3.addRow(yk.neg().getRow(i));

            z2 = z2.addRow(0, Matrix.ones(1, 1));

            var delta2 = Theta[1].transpose().mmul(delta3).mulM(
                            z2.apply(Util.sigmoidGradient));
            delta2 = delta2.removeColumn(0);

            grad[0] += (delta2.mmul(a1));
            grad[1] += (delta3.mmul(a2));
        }

        grad[0] *= (1 / m);
        grad[1] *= (1 / m);

        // apply regularization term to the gradient

        return {cost: cost, grad: grad};
    }

    function randomInitialzeTheta(labelsIn, labelsOut) {
        var epsilonRange = 0.12; // values around

        return Matrix.rand(labelsOut, labelsIn).mulS(2).mulS(epsilonRange).addS(-epsilonRange);
    }

    this.train = function(XArg, yArg, learningRate, lambdaArg, numberOfLabels, iterations, hiddenLayerSize) {
        X = XArg;
        y = yArg;
        var m = XArg.rows;

        Theta[0] = randomInitialzeTheta(XArg, hiddenLayerSize); // TODO: be careful
        Theta[1] = randomInitialzeTheta(hiddenLayerSize, numberOfLabels);

        var H = Theta.transpose().mmul(X.transpose());
        var result;

        for(var i = 0; i < iterations; ++i) {
            result = costFunction(lambdaArg, numberOfLabels);
            Theta[0] = Theta[0].addS((alpha / m) * result.grad[0]);
            Theta[1] = Theta[1].addS((alpha / m) * result.grad[1]);
        }
    };

    this.predict = function(X) {
        var m = X.rows;

        var predictions = Matrix.zeros(m, 1);

        var h1 = X.addColumn(0, Matrix.ones(m, 1)).mmul(Theta[0].transpose()).apply(Util.sigmoid);
        var h2 = h1.addColumn(0, Matrix.ones(m, 1)).mmul(Theta[1].transpose()).apply(Util.sigmoid);

        return h2.maxRowIndex(0);
    };

};
