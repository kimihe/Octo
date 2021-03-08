# coding: utf-8
import time, sys
from common.functions import *
from common.util import im2col, col2im
from quantizer.KMQuantizer_v3 import KMConvQuantizer, KMAffineQuantizer
from quantizer.KMQATScheme import KMQATScheme


class Relu:
    def __init__(self, layer_id='relu'):
        self.mask = None
        self.layer_id = layer_id

    def forward(self, x):

        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx


class Affine:
    def __init__(self, W, b, enable_fp_qat=False, enable_bp_gradient_quantization=False, layer_id='Affine'):
        self.W = W
        self.b = b
        self.enable_fp_qat = enable_fp_qat
        self.enable_bp_gradient_quantization = enable_bp_gradient_quantization
        self.layer_id = layer_id
        
        self.x = None
        self.original_x_shape = None

        self.dW = None
        self.db = None

        self.quantizer = KMAffineQuantizer()
        self.fp_W_int8 = None

    def forward(self, x):
        if self.enable_fp_qat:

            self.original_x_shape = x.shape
            x = x.reshape(x.shape[0], -1)
            self.x = x

            x_int8 = self.quantizer.affine_x_quantize(self.x)
            W_int8 = self.quantizer.affine_W_quantize(self.W)

            out_int8 = np.dot(x_int8, W_int8)
            out = self.quantizer.affine_dequantize(out_int8)
            out = out + self.b

            self.x = x_int8
            self.fp_W_int8 = W_int8

            return out

        else:
            self.original_x_shape = x.shape
            x = x.reshape(x.shape[0], -1)
            self.x = x

            out = np.dot(self.x, self.W) + self.b

            return out

    def backward(self, dout):
        if self.enable_bp_gradient_quantization:
            dout_int8 = self.quantizer.bp_dout_quantize(dout)

            dx_int8 = np.dot(dout_int8, self.fp_W_int8.T)
            dx_q = self.quantizer.bp_dx_dequantize_memory_saving(dx_int8)
            dx = dx_q.reshape(*self.original_x_shape)

            dW_int8 = np.dot(self.x.T, dout_int8)
            dW_q = self.quantizer.bp_dW_dequantize_memory_saving(dW_int8)
            self.dW = dW_q

            db = np.sum(dout, axis=0)
            db_int8 = self.quantizer.fake_quantize(db)
            db_q = self.quantizer.fake_dequantize(db_int8)
            self.db = db_q

            return dx

        else:
            dx = np.dot(dout, self.W.T)
            self.dW = np.dot(self.x.T, dout)
            self.db = np.sum(dout, axis=0)

            dx = dx.reshape(*self.original_x_shape)

            return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        
        return dx


class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask


class BatchNormalization:
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None, layer_id='BatchNorm'):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None
        self.layer_id = layer_id

        self.running_mean = running_mean
        self.running_var = running_var  

        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, train_flg)
        
        return out.reshape(*self.input_shape)
            
    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)
                        
        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std
            
            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * var            
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))
            
        out = self.gamma * xn + self.beta 
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)

        return dx

    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size
        
        self.dgamma = dgamma
        self.dbeta = dbeta
        
        return dx


class Convolution:
    def __init__(self, W, b, stride=1, pad=0, enable_fp_qat=False, enable_bp_gradient_quantization=False,
                 enable_x_clipping=False, enable_W_clipping=False,
                 enable_gradient_clipping=True,
                 layer_id='Conv', qat_scheme=KMQATScheme.Dequantization):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        self.enable_fp_qat = enable_fp_qat
        self.enable_bp_gradient_quantization = enable_bp_gradient_quantization
        self.layer_id = layer_id

        self.x = None   
        self.col = None
        self.col_W = None

        self.dW = None
        self.db = None

        self.quantizer = KMConvQuantizer()
        self.qat_scheme = qat_scheme

        self.enable_x_clipping = enable_x_clipping
        self.enable_W_clipping = enable_W_clipping
        self.enable_gradient_clipping = enable_gradient_clipping
        # self.enable_compensation_clipping = enable_compensation_clipping



    def parametrized_range_clipping(self, tensor, z=2.576):
        # 95%: 1.96; 99%: 2.576
        n = tensor.size
        if n == 0:
            return tensor

        clip_max = 0.95 * np.max(tensor)
        clip_min = 0.95 * np.min(tensor)

        return np.clip(tensor, clip_min, clip_max)

    def forward(self, x):
        if self.enable_fp_qat:
            if self.qat_scheme == KMQATScheme.Dequantization:
                x_int8 = self.quantizer.conv2d_x_quantize(x)
                W_int8 = self.quantizer.conv2d_W_quantize(self.W)

                FN, C, FH, FW = W_int8.shape
                N, C, H, W = x_int8.shape
                out_h = 1 + int((H + 2 * self.pad - FH) / self.stride)
                out_w = 1 + int((W + 2 * self.pad - FW) / self.stride)

                col_int8 = im2col(x_int8, FH, FW, self.stride, self.pad)
                col_W_int8 = W_int8.reshape(FN, -1).T

                out_int8 = np.dot(col_int8, col_W_int8)
                out = self.quantizer.conv2d_dequantize(out_int8)
                out = out + self.b

                out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

                col_q = self.quantizer.conv2d_x_dequantize(col_int8)
                col_W_q = self.quantizer.conv2d_W_dequantize(col_W_int8)

                x_q = self.quantizer.conv2d_x_dequantize(x_int8)
                W_q = self.quantizer.conv2d_W_dequantize(W_int8)

                self.x = x_q
                self.col = col_q
                self.col_W = col_W_q

                return out

            elif self.qat_scheme == KMQATScheme.ErrorCompensation:
                x_int8 = self.quantizer.conv2d_x_quantize(x)
                W_int8 = self.quantizer.conv2d_W_quantize(self.W)

                FN, C, FH, FW = W_int8.shape
                N, C, H, W = x_int8.shape
                out_h = 1 + int((H + 2 * self.pad - FH) / self.stride)
                out_w = 1 + int((W + 2 * self.pad - FW) / self.stride)

                col_int8 = im2col(x_int8, FH, FW, self.stride, self.pad)
                col_W_int8 = W_int8.reshape(FN, -1).T

                col_x = im2col(x, FH, FW, self.stride, self.pad)
                col_W = self.W.reshape(FN, -1).T
                compensation = self.quantizer.compensation(col_W, col_x, col_W_int8, col_int8)

                out_int8 = np.dot(col_int8, col_W_int8)
                out = self.quantizer.conv2d_dequantize(out_int8 + compensation)  # compensate the out_int8
                out = out + self.b

                out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

                col_q = self.quantizer.conv2d_x_dequantize(col_int8)
                col_W_q = self.quantizer.conv2d_W_dequantize(col_W_int8)

                x_q = self.quantizer.conv2d_x_dequantize(x_int8)
                W_q = self.quantizer.conv2d_W_dequantize(W_int8)

                self.x = x_q
                self.col = col_q
                self.col_W = col_W_q

                self.W = W_q

                return out

            elif self.qat_scheme == KMQATScheme.LossAwareCompensation:

                if self.enable_x_clipping:
                    x_clipped = self.parametrized_range_clipping(x)
                    x_int8 = self.quantizer.conv2d_x_quantize(x_clipped)
                else:
                    x_int8 = self.quantizer.conv2d_x_quantize(x)

                if self.enable_W_clipping:
                    W_clipped = self.parametrized_range_clipping(self.W)
                    W_int8 = self.quantizer.conv2d_W_quantize(W_clipped)
                else:
                    W_int8 = self.quantizer.conv2d_W_quantize(self.W)


                FN, C, FH, FW = W_int8.shape
                N, C, H, W = x_int8.shape
                out_h = 1 + int((H + 2 * self.pad - FH) / self.stride)
                out_w = 1 + int((W + 2 * self.pad - FW) / self.stride)

                col_int8 = im2col(x_int8, FH, FW, self.stride, self.pad)
                col_W_int8 = W_int8.reshape(FN, -1).T

                out_int8 = np.dot(col_int8, col_W_int8)
                out = self.quantizer.conv2d_dequantize(out_int8)
                out = out + self.b

                out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

                col_q = self.quantizer.conv2d_x_dequantize(col_int8)
                col_W_q = self.quantizer.conv2d_W_dequantize(col_W_int8)

                x_q = self.quantizer.conv2d_x_dequantize(x_int8)
                W_q = self.quantizer.conv2d_W_dequantize(W_int8)

                col = im2col(x, FH, FW, self.stride, self.pad)
                col_W = self.W.reshape(FN, -1).T
                self.x = x
                self.col = col
                self.col_W = col_W

                return out

            else:
                print("====== Exception: Undefined QAT Scheme in {} {}".
                      format(self.layer_id, sys._getframe().f_code.co_name))
                pass

        else:
            FN, C, FH, FW = self.W.shape
            N, C, H, W = x.shape
            out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
            out_w = 1 + int((W + 2*self.pad - FW) / self.stride)

            col = im2col(x, FH, FW, self.stride, self.pad)
            col_W = self.W.reshape(FN, -1).T

            out = np.dot(col, col_W) + self.b
            out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

            self.x = x
            self.col = col
            self.col_W = col_W

            return out

    def backward(self, dout):
        if self.enable_bp_gradient_quantization:
            if self.qat_scheme == KMQATScheme.Dequantization:

                FN, C, FH, FW = self.W.shape
                dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

                db = np.sum(dout, axis=0)
                db_int8 = self.quantizer.fake_quantize(db)
                db_q = self.quantizer.fake_dequantize(db_int8)
                self.db = db_q

                col_int8 = self.quantizer.bp_col_quantize(self.col)
                dout_int8 = self.quantizer.bp_dout_quantize(dout)
                dW_int8 = np.dot(col_int8.T, dout_int8)
                dW_q = self.quantizer.bp_dW_dequantize(dW_int8)
                dW_q = dW_q.transpose(1, 0).reshape(FN, C, FH, FW)
                self.dW = dW_q

                dcol_W_int8 = self.quantizer.bp_col_W_quantize(self.col_W)
                dcol_int8 = np.dot(dout_int8, dcol_W_int8.T)
                dcol_q = self.quantizer.bp_dcol_dequantize(dcol_int8)
                dx = col2im(dcol_q, self.x.shape, FH, FW, self.stride, self.pad)

                return dx

            elif self.qat_scheme == KMQATScheme.ErrorCompensation:
                FN, C, FH, FW = self.W.shape
                dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

                db = np.sum(dout, axis=0)
                db_int8 = self.quantizer.fake_quantize(db)
                db_q = self.quantizer.fake_dequantize(db_int8)
                self.db = db_q

                col_int8 = self.quantizer.bp_col_quantize(self.col)
                dout_int8 = self.quantizer.bp_dout_quantize(dout)
                dW_int8 = np.dot(col_int8.T, dout_int8)
                compensation = self.quantizer.bp_compensation(self.col, col_int8, dout_int8)
                dW_q = self.quantizer.bp_dW_dequantize(dW_int8 + compensation)
                dW_q = dW_q.transpose(1, 0).reshape(FN, C, FH, FW)
                self.dW = dW_q

                dcol_W_int8 = self.quantizer.bp_col_W_quantize(self.col_W)
                dcol_int8 = np.dot(dout_int8, dcol_W_int8.T)
                compensation_dx = self.quantizer.bp_compensation(self.col_W, dcol_W_int8, dout_int8, com_dx=True)
                dcol_q = self.quantizer.bp_dcol_dequantize(dcol_int8 + compensation_dx)
                dx = col2im(dcol_q, self.x.shape, FH, FW, self.stride, self.pad)

                return dx

            elif self.qat_scheme == KMQATScheme.LossAwareCompensation:
                FN, C, FH, FW = self.W.shape
                dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)


                db = np.sum(dout, axis=0)
                db_int8 = self.quantizer.fake_quantize(db)
                db_q = self.quantizer.fake_dequantize(db_int8)
                self.db = db_q

                if self.enable_gradient_clipping:
                    col_clipped = self.parametrized_range_clipping(self.col)
                    dout_clipped = self.parametrized_range_clipping(dout)

                    col_int8 = self.quantizer.bp_col_quantize(col_clipped)
                    dout_int8 = self.quantizer.bp_dout_quantize(dout_clipped)
                else:
                    col_int8 = self.quantizer.bp_col_quantize(self.col)
                    dout_int8 = self.quantizer.bp_dout_quantize(dout)

                dW_int8 = np.dot(col_int8.T, dout_int8)
                dW_q = self.quantizer.bp_dW_dequantize(dW_int8)
                dW_q = dW_q.transpose(1, 0).reshape(FN, C, FH, FW)
                self.dW = dW_q

                if self.enable_gradient_clipping:
                    col_W_clipped = self.parametrized_range_clipping(self.col_W)
                    dcol_W_int8 = self.quantizer.bp_col_W_quantize(col_W_clipped)
                else:
                    dcol_W_int8 = self.quantizer.bp_col_W_quantize(self.col_W)

                dcol_int8 = np.dot(dout_int8, dcol_W_int8.T)
                dcol_q = self.quantizer.bp_dcol_dequantize(dcol_int8)
                dx = col2im(dcol_q, self.x.shape, FH, FW, self.stride, self.pad)

                return dx

            else:
                print("====== Exception: Undefined QAT Scheme in {} {}".
                      format(self.layer_id, sys._getframe().f_code.co_name))
                pass

        else:
            if self.qat_scheme == KMQATScheme.LossAwareCompensation:
                FN, C, FH, FW = self.W.shape
                dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

                self.db = np.sum(dout, axis=0)
                self.dW = np.dot(self.col.T, dout)
                self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

                dcol = np.dot(dout, self.col_W.T)
                dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

                return dx

            else:
                FN, C, FH, FW = self.W.shape
                dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

                self.db = np.sum(dout, axis=0)
                self.dW = np.dot(self.col.T, dout)
                self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

                dcol = np.dot(dout, self.col_W.T)
                dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

                return dx


class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        
        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,)) 
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        
        return dx


class Compensation:
    def __init__(self, alpha, mu, offset, enable_compensation_clipping=False, layer_id='Compensation'):
        self.alpha = alpha
        self.mu = mu
        self.offset = offset
        self.enable_compensation_clipping = enable_compensation_clipping
        self.layer_id = layer_id

        self.d_alpha = None
        self.d_mu = None
        self.d_offset = None

        self.clip_max = None
        self.clip_min = None
        self.fp_out_before_clipping = None

    def inspection(self):
        path = "../debug/{}.npy".format(sys._getframe().f_code.co_name)
        np.save(path, self.alpha * self.mu + self.offset)

    def parametrized_range_clipping(self, tensor, z=2.576):
        n = tensor.size
        if n == 0:
            return tensor

        self.clip_max = 0.95 * np.max(tensor)
        self.clip_min = 0.95 * np.min(tensor)

        return np.clip(tensor, self.clip_min, self.clip_max)

    def forward(self, x):
        out = x + self.alpha * self.mu + self.offset
        self.fp_out_before_clipping = out

        if self.enable_compensation_clipping:
            out = self.parametrized_range_clipping(out)

        return out

    def backward(self, dout):
        if self.enable_compensation_clipping:  # TODO: low code efficiency
            dout_clip_mask = self.fp_out_before_clipping
            with np.nditer(dout_clip_mask, op_flags=['readwrite']) as it:
                for x in it:
                    if x > self.clip_max or x < self.clip_min:
                        x[...] = 0.0
                    else:
                        x[...] = 1.0
            dout *= dout_clip_mask

        self.d_alpha = np.sum(dout * self.mu, axis=0)

        self.d_mu = self.alpha * dout

        self.d_offset = dout
        dx = dout

        return dx
