function logistic(val) {
  return 1 / (1 + Math.exp(-val));
}

function expELU(val, param) {
  return val < 0 ? param * (Math.exp(val) - 1) : val;
}

function softExponential(val, param) {
  if (param < 0) {
    return -Math.log(1 - param * (val + param)) / param;
  }
  if (param > 0) {
    return ((Math.exp(param * val) - 1) / param) + param;
  }
  return val;
}

function softExponentialPrime(val, param) {
  if (param < 0) {
    return 1 / (1 - param * (param + val));
  } else {
    return Math.exp(param * val);
  }
}

const ACTIVATION_FUNCTIONS = {
  tanh: {
    activation: Math.tanh,
    derivate: (val) => 1 - (val * val)
  },
  identity: {
    activation: (val) => val,
    derivate: () => 1
  },
  logistic: {
    activation: logistic,
    derivate: (val) => logistic(val) * (1 - logistic(val))
  },
  arctan: {
    activation: Math.atan,
    derivate: (val) => 1 / (val * val + 1)
  },
  softsign: {
    activation: (val) => val / (1 + Math.abs(val)),
    derivate: (val) => 1 / ((1 + Math.abs(val)) * (1 + Math.abs(val)))
  },
  relu: {
    activation: (val) => (val < 0 ? 0 : val),
    derivate: (val) => (val < 0 ? 0 : 1)
  },
  softplus: {
    activation: (val) => Math.log(1 + Math.exp(val)),
    derivate: (val) => 1 / (1 + Math.exp(-val))
  },
  bent: {
    activation: (val) => ((Math.sqrt(val * val + 1) - 1) / 2) + val,
    derivate: (val) => (val / (2 * Math.sqrt(val * val + 1))) + 1
  },
  sinusoid: {
    activation: Math.sin,
    derivate: Math.cos
  },
  sinc: {
    activation: (val) => (val === 0 ? 1 : Math.sin(val) / val),
    derivate: (val) => (val === 0 ? 0 : (Math.cos(val) / val) - (Math.sin(val) / (val * val)))
  },
  gaussian: {
    activation: (val) => Math.exp(-(val * val)),
    derivate: (val) => -2 * val * Math.exp(-(val * val))
  },
  'parametric-relu': {
    activation: (val, param) => (val < 0 ? param * val : val),
    derivate: (val, param) => (val < 0 ? param : 1)
  },
  'exponential-elu': {
    activation: expELU,
    derivate: (val, param) => (val < 0 ? expELU(val, param) + param : 1)
  },
  'soft-exponential': {
    activation: softExponential,
    derivate: softExponentialPrime
  }
};

export default ACTIVATION_FUNCTIONS;
