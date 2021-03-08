import numpy as np

def probabilistic_round(x):
    # return int(math.floor(x + random.random()))

    res = np.floor(x + np.random.random(x.shape))
    res = res.astype(np.int)
    return res


class KMConvQuantizer:

    def __init__(self):
        self.x_scale = 1.0
        self.W_scale = 1.0

        self.x_zero_point = 0.0
        self.W_zero_point = 0.0

        self.dout_scale = 1.0
        self.col_scale = 1.0
        self.col_W_scale = 1.0

        self.dout_zero_point = 0.0
        self.col_zero_point = 0.0
        self.col_W_zero_point = 0.0

        self.fake_scale = 1.0
        self.fake_zero_point = 0.0

    def calcScaleZeroPoint(self, min_val, max_val, num_bits=8):
        qmin = 0.
        qmax = 2. ** num_bits - 1.
        zero_point = 0

        scale = (max_val - min_val) / (qmax - qmin)

        initial_zero_point = qmin - min_val / scale

        if initial_zero_point < qmin:
            zero_point = qmin
        elif initial_zero_point > qmax:
            zero_point = qmax
        else:
            zero_point = initial_zero_point

        zero_point = 0
        zero_point = int(zero_point)

        return scale, zero_point

    def calcScaleZeroPointSymm(self, min_val, max_val, num_bits=8):
        qmax = 2 ** (num_bits-1) - 1
        qmin = -qmax

        scale = (max_val - min_val) / (qmax - qmin)
        zero_point = 0.0

        if scale == 0.0:
            scale = 1.0

        return scale, zero_point

    def calcScaleZeroPointAsymm(self, min_val, max_val, num_bits=8):
        qmax = 2. ** num_bits - 1.
        qmin = -qmax

        scale = (max_val - min_val) / (qmax - qmin)
        if scale == 0.0:
            scale = 1.0
            zero_point = 0.0
        else:
            zero_point = qmax - max_val / scale

            if zero_point < qmin:
                zero_point = qmin

            if zero_point > qmax:
                zero_point = qmax

        return scale, zero_point

    def calcScaleZeroPointSymmErrorCompensation(self, min_val, max_val, num_bits=8):
        qmax = 2 ** (num_bits - 1) - 1
        qmin = -qmax

        scale = (max_val - min_val) / (qmax - qmin)
        zero_point = qmax - max_val / scale

        return scale, zero_point

    def compensation(self, w_fp32, x_fp32, w_int8, x_int8):
        self.W_scale, self.W_zero_point = self.calcScaleZeroPointSymmErrorCompensation(min_val=w_fp32.min(),
                                                                      max_val=w_fp32.max(),
                                                                      num_bits=8)
        self.x_scale, self.x_zero_point = self.calcScaleZeroPointSymmErrorCompensation(min_val=x_fp32.min(),
                                                                      max_val=x_fp32.max(),
                                                                      num_bits=8)
        delta_w, delta_x = w_fp32 / self.W_scale - w_int8, x_fp32 / self.x_scale - x_int8
        return np.dot(delta_x, delta_w) + np.dot(x_int8, delta_w) + np.dot(delta_x, w_int8)

    def bp_compensation(self, tensor, tensor_int8, dout, com_dx=False):
        if com_dx:
            delta_w = tensor / self.col_W_scale - tensor_int8
            compensation = np.dot(dout, delta_w.T)
        else:
            delta_x = tensor / self.col_scale - tensor_int8
            compensation = np.dot(delta_x.T, dout)

        return compensation

    def conv2d_x_quantize(self, tensor_fp32):

        self.x_scale, self.x_zero_point = self.calcScaleZeroPointAsymm(min_val=tensor_fp32.min(),
                                                              max_val=tensor_fp32.max(),
                                                              num_bits=8)

        tensor_int8 = probabilistic_round(tensor_fp32 / self.x_scale + self.x_zero_point)
        return tensor_int8

    def conv2d_W_quantize(self, tensor_fp32):

        self.W_scale, self.W_zero_point = self.calcScaleZeroPointAsymm(min_val=tensor_fp32.min(),
                                                              max_val=tensor_fp32.max(),
                                                              num_bits=8)

        tensor_int8 = probabilistic_round(tensor_fp32 / self.W_scale + self.W_zero_point)
        return tensor_int8

    def conv2d_x_dequantize(self, tensor_int8):

        tensor_fp32 = (tensor_int8 - self.x_zero_point) * self.x_scale
        return tensor_fp32

    def conv2d_W_dequantize(self, tensor_int8):

        tensor_fp32 = (tensor_int8 - self.W_zero_point) * self.W_scale
        return tensor_fp32

    def conv2d_dequantize(self, tensor_int8):

        tensor_fp32 = tensor_int8 * (self.x_scale * self.W_scale)
        return tensor_fp32

    def bp_dout_quantize(self, tensor_fp32):
        self.dout_scale, self.dout_zero_point = self.calcScaleZeroPointSymm(min_val=tensor_fp32.min(),
                                                                      max_val=tensor_fp32.max(),
                                                                      num_bits=8)

        tensor_int8 = np.around(tensor_fp32 / self.dout_scale)
        return tensor_int8

    def bp_dout_dequantize(self, tensor_int8):
        tensor_fp32 = tensor_int8 * self.dout_scale
        return tensor_fp32

    def bp_col_quantize(self, tensor_fp32):
        self.col_scale, self.col_zero_point = self.calcScaleZeroPointSymm(min_val=tensor_fp32.min(),
                                                                            max_val=tensor_fp32.max(),
                                                                            num_bits=8)

        tensor_int8 = np.around(tensor_fp32 / self.col_scale)
        return tensor_int8

    def bp_col_dequantize(self, tensor_int8):
        tensor_fp32 = tensor_int8 * self.col_scale
        return tensor_fp32

    def bp_col_W_quantize(self, tensor_fp32):
        self.col_W_scale, self.col_W_zero_point = self.calcScaleZeroPointSymm(min_val=tensor_fp32.min(),
                                                                          max_val=tensor_fp32.max(),
                                                                          num_bits=8)

        tensor_int8 = np.around(tensor_fp32 / self.col_W_scale)
        return tensor_int8

    def bp_col_W_dequantize(self, tensor_int8):
        tensor_fp32 = tensor_int8 * self.col_W_scale
        return tensor_fp32

    def bp_dW_dequantize(self, tensor_int8):
        tensor_fp32 = tensor_int8 * (self.col_scale * self.dout_scale)
        return tensor_fp32

    def bp_dcol_dequantize(self, tensor_int8):
        tensor_fp32 = tensor_int8 * (self.dout_scale * self.col_W_scale)
        return tensor_fp32

    def fake_quantize(self, tensor_fp32):
        self.fake_scale, self.fake_zero_point = self.calcScaleZeroPointSymm(min_val=tensor_fp32.min(),
                                                              max_val=tensor_fp32.max(),
                                                              num_bits=8)

        tensor_int8 = np.around(tensor_fp32 / self.fake_scale)
        return tensor_int8

    def fake_dequantize(self, tensor_int8):
        tensor_fp32 = tensor_int8 * self.fake_scale
        return tensor_fp32


class KMAffineQuantizer:

    def __init__(self):
        self.x_scale = 1.0
        self.W_scale = 1.0

        self.x_zero_point = 0.0
        self.W_zero_point = 0.0

        self.bp_dout_scale = 1.0
        self.bp_W_scale = 1.0
        self.bp_x_scale = 1.0

        self.bp_dout_zero_point = 0.0
        self.bp_W_zero_point = 0.0
        self.bp_x_zero_point = 0.0

        self.fake_scale = 1.0
        self.fake_zero_point = 0.0

    def calcScaleZeroPoint(self, min_val, max_val, num_bits=8):
        qmin = 0.
        qmax = 2. ** num_bits - 1.
        zero_point = 0

        scale = (max_val - min_val) / (qmax - qmin)

        initial_zero_point = qmin - min_val / scale

        if initial_zero_point < qmin:
            zero_point = qmin
        elif initial_zero_point > qmax:
            zero_point = qmax
        else:
            zero_point = initial_zero_point

        zero_point = 0
        zero_point = int(zero_point)

        return scale, zero_point

    def calcScaleZeroPointSymm(self, min_val, max_val, num_bits=8):
        qmax = 2 ** (num_bits-1) - 1
        qmin = -qmax

        scale = (max_val - min_val) / (qmax - qmin)
        zero_point = 0

        return scale, zero_point

    def calcScaleZeroPointAsymm(self, min_val, max_val, num_bits=8):
        qmax = 2. ** num_bits - 1.
        qmin = -qmax

        scale = 1.0
        zero_point = 0.0

        scale = (max_val - min_val) / (qmax - qmin)
        if scale == 0.0:
            scale = 1.0
            zero_point = 0.0
        else:
            zero_point = qmax - max_val / scale

            if zero_point < qmin:
                zero_point = qmin

            if zero_point > qmax:
                zero_point = qmax

        return scale, zero_point

    def affine_x_quantize(self, tensor_fp32):

        self.x_scale, self.x_zero_point = self.calcScaleZeroPointSymm(min_val=tensor_fp32.min(),
                                                                  max_val=tensor_fp32.max(),
                                                                  num_bits=8)

        tensor_int8 = np.around(tensor_fp32 / self.x_scale)
        return tensor_int8

    def affine_W_quantize(self, tensor_fp32):

        self.W_scale, self.W_zero_point = self.calcScaleZeroPointSymm(min_val=tensor_fp32.min(),
                                                                  max_val=tensor_fp32.max(),
                                                                  num_bits=8)

        tensor_int8 = np.around(tensor_fp32 / self.W_scale)
        return tensor_int8

    def affine_x_dequantize(self, tensor_int8):

        tensor_fp32 = tensor_int8 * self.x_scale
        return tensor_fp32

    def affine_W_dequantize(self, tensor_int8):

        tensor_fp32 = tensor_int8 * self.W_scale
        return tensor_fp32

    def affine_dequantize(self, tensor_int8):

        tensor_fp32 = tensor_int8 * (self.x_scale * self.W_scale)
        return tensor_fp32

    def bp_dout_quantize(self, tensor_fp32):
        self.bp_dout_scale, self.bp_dout_zero_point = self.calcScaleZeroPointSymm(min_val=tensor_fp32.min(),
                                                                            max_val=tensor_fp32.max(),
                                                                            num_bits=8)

        tensor_int8 = np.around(tensor_fp32 / self.bp_dout_scale)
        return tensor_int8

    def bp_dout_dequantize(self, tensor_int8):
        tensor_fp32 = tensor_int8 * self.bp_dout_scale
        return tensor_fp32

    def bp_W_quantize(self, tensor_fp32):
        self.bp_W_scale, self.bp_W_zero_point_ = self.calcScaleZeroPointSymm(min_val=tensor_fp32.min(),
                                                                            max_val=tensor_fp32.max(),
                                                                            num_bits=8)

        tensor_int8 = np.around(tensor_fp32 / self.bp_W_scale)
        return tensor_int8

    def bp_W_dequantize(self, tensor_int8):
        tensor_fp32 = tensor_int8 * self.bp_W_scale
        return tensor_fp32

    def bp_x_quantize(self, tensor_fp32):
        self.bp_x_scale, self.bp_x_zero_point = self.calcScaleZeroPointSymm(min_val=tensor_fp32.min(),
                                                                            max_val=tensor_fp32.max(),
                                                                            num_bits=8)

        tensor_int8 = np.around(tensor_fp32 / self.bp_x_scale)
        return tensor_int8

    def bp_x_dequantize(self, tensor_int8):
        tensor_fp32 = tensor_int8 * self.bp_x_scale
        return tensor_fp32

    def bp_dx_dequantize(self, tensor_int8):
        tensor_fp32 = tensor_int8 * (self.bp_dout_scale * self.bp_W_scale)
        return tensor_fp32

    def bp_dx_dequantize_memory_saving(self, tensor_int8):
        tensor_fp32 = tensor_int8 * (self.bp_dout_scale * self.W_scale)
        return tensor_fp32

    def bp_dW_dequantize(self, tensor_int8):
        tensor_fp32 = tensor_int8 * (self.bp_x_scale * self.bp_dout_scale)
        return tensor_fp32

    def bp_dW_dequantize_memory_saving(self, tensor_int8):
        tensor_fp32 = tensor_int8 * (self.x_scale * self.bp_dout_scale)
        return tensor_fp32

    def fake_quantize(self, tensor_fp32):
        self.fake_scale, self.fake_zero_point = self.calcScaleZeroPointSymm(min_val=tensor_fp32.min(),
                                                                        max_val=tensor_fp32.max(),
                                                                        num_bits=8)

        tensor_int8 = np.around(tensor_fp32 / self.fake_scale)
        return tensor_int8

    def fake_dequantize(self, tensor_int8):
        tensor_fp32 = tensor_int8 * self.fake_scale
        return tensor_fp32