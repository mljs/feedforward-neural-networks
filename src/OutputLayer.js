import { Layer } from './Layer';

export class OutputLayer extends Layer {
  constructor(options) {
    super(options);

    this.activationFunction = function (i, j) {
      this.set(i, j, Math.exp(this.get(i, j)));
    };
  }

  static load(model) {
    if (model.model !== 'Layer') {
      throw new RangeError('the current model is not a Layer model');
    }

    return new OutputLayer(model);
  }
}
