Util = {
    sigmoid : function(value) {
        return 1.0 / (1 + Math.exp(-value));
    },

    sigmoidGradient : function(value) {
        return this.sigmoid(value) * (1 - this.sigmoid(value));
    }
};