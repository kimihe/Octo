# coding: utf-8
import sys, os
sys.path.append(os.pardir)
from collections import OrderedDict
from common.layers import *

class SimpleConvNet:
    """1 Conv, 2 Affines, 8 layers in total

    网络结构如下所示:
        conv1 - (compensation1 - batchnorm1) - relu_of_conv_1 - pool1
        affine1 - relu_of_affine1
        affine2
        softmax
    """

    def __init__(self, input_dim=(1, 28, 28),
                 conv_param_1={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1,
                             'enable_fp_qat': False, 'enable_bp_gradient_quantization': False,
                             'qat_scheme': KMQATScheme.FullPrecision},
                 affine1_param={'enable_fp_qat': False, 'enable_bp_gradient_quantization': False},
                 affine2_param={'enable_fp_qat': False, 'enable_bp_gradient_quantization': False},
                 hidden_size=100, output_size=10, weight_init_std=0.01,
                 enable_compensation_L2_regularization=False, compensation_L2_regularization_lambda=0.1,
                 mini_batch_size=100):

        self.network_name = type(self).__name__
        self.conv_param_list = [conv_param_1]
        self.affine_param_list = [affine1_param, affine2_param]
        self.enable_compensation_L2_regularization = enable_compensation_L2_regularization
        self.compensation_L2_regularization_lambda = compensation_L2_regularization_lambda
        self.mini_batch_size = mini_batch_size

        for i, v in enumerate(self.conv_param_list):
            if v['qat_scheme'] == KMQATScheme.FullPrecision:
                v['enable_fp_qat'] = False
                v['enable_bp_gradient_quantization'] = False

        self.network_config = {}
        for i, v in enumerate(self.conv_param_list):
            self.network_config['Conv_param_' + str(i + 1)] = v

        for i, v in enumerate(self.affine_param_list):
            self.network_config['Affine_param_' + str(i + 1)] = v

        filter_num = conv_param_1['filter_num']
        filter_size = conv_param_1['filter_size']
        filter_pad = conv_param_1['pad']
        filter_stride = conv_param_1['stride']
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2*filter_pad) / filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2))

        self.params = {}
        self.params['Conv_W_1'] = weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['Conv_b_1'] = np.zeros(filter_num)
        self.params['Affine_W_1'] = weight_init_std * np.random.randn(pool_output_size, hidden_size)
        self.params['Affine_b_1'] = np.zeros(hidden_size)
        self.params['Affine_W_2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['Affine_b_2'] = np.zeros(output_size)

        compensation_param_shape_list = [(self.mini_batch_size, 30, 24, 24)]
        batchnorm_param_shape_list = []
        for i in range(len(compensation_param_shape_list)):
            ele = (compensation_param_shape_list[i][0],
                   compensation_param_shape_list[i][1] * compensation_param_shape_list[i][2] * compensation_param_shape_list[i][3])
            batchnorm_param_shape_list.append(ele)

        self.layers = OrderedDict()
        for i in range(len(self.conv_param_list)):
            self.layers['Conv_' + str(i+1)] = Convolution(self.params['Conv_W_' + str(i+1)],
                                                          self.params['Conv_b_' + str(i+1)],
                                                          self.network_config['Conv_param_' + str(i+1)]['stride'],
                                                          self.network_config['Conv_param_' + str(i+1)]['pad'],
                                                          enable_fp_qat=self.network_config['Conv_param_' + str(i+1)]['enable_fp_qat'],
                                                          enable_bp_gradient_quantization=self.network_config['Conv_param_' + str(i+1)]['enable_bp_gradient_quantization'],
                                                          layer_id='Conv_' + str(i+1),
                                                          qat_scheme=self.network_config['Conv_param_' + str(i+1)]['qat_scheme'])

            if self.network_config['Conv_param_' + str(i+1)]['qat_scheme'] == KMQATScheme.LossAwareCompensation:
                self.params['Compensation_alpha_' + str(i+1)] = np.ones(compensation_param_shape_list[i], dtype=np.float64)
                self.params['Compensation_mu_' + str(i+1)] = 1e-3 * np.random.normal(0.0, 1.0, size=compensation_param_shape_list[i])
                self.params['Compensation_offset_' + str(i+1)] = 1e-3 * np.random.normal(0.0, 1.0, size=compensation_param_shape_list[i])
                self.layers['Compensation_' + str(i+1)] = Compensation(self.params['Compensation_alpha_' + str(i+1)],
                                                                       self.params['Compensation_mu_' + str(i+1)],
                                                                       self.params['Compensation_offset_' + str(i+1)],
                                                                       layer_id='Compensation_' + str(i+1))

                self.params['BatchNorm_gamma_' + str(i+1)] = np.ones(batchnorm_param_shape_list[i])
                self.params['BatchNorm_beta_' + str(i+1)] = np.zeros(batchnorm_param_shape_list[i])
                self.layers['BatchNorm_' + str(i+1)] = BatchNormalization(self.params['BatchNorm_gamma_' + str(i+1)],
                                                                          self.params['BatchNorm_beta_' + str(i+1)],
                                                                          layer_id='BatchNorm_' + str(i+1))

            self.layers['Relu_of_Conv_' + str(i+1)] = Relu()

            if i == 0:
                self.layers['Pool_' + str(i+1)] = Pooling(pool_h=2, pool_w=2, stride=2)

        for i in range(len(self.affine_param_list)):
            self.layers['Affine_' + str(i+1)] = Affine(self.params['Affine_W_' + str(i+1)],
                                                       self.params['Affine_b_' + str(i+1)],
                                                       enable_fp_qat=self.network_config['Affine_param_' + str(i+1)]['enable_fp_qat'],
                                                       enable_bp_gradient_quantization=self.network_config['Affine_param_' + str(i+1)]['enable_bp_gradient_quantization'],
                                                       layer_id='Affine_' + str(i+1))

            if i == 0:
                self.layers['Relu_of_Affine_' + str(i+1)] = Relu()

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        if self.enable_compensation_L2_regularization:
            y = self.predict(x)

            l2_regularization = 0
            for i in range(len(self.conv_param_list)):
                if self.network_config['Conv_param_' + str(i+1)]['qat_scheme'] == KMQATScheme.LossAwareCompensation:
                    mu = self.params['Compensation_mu_' + str(i+1)]
                    offset = self.params['Compensation_offset_' + str(i+1)]
                    l2_regularization += 0.5 * self.compensation_L2_regularization_lambda * (np.sum(mu ** 2) + np.sum(offset ** 2))
            return self.last_layer.forward(y, t) + l2_regularization
        else:
            y = self.predict(x)
            return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        batch_size = self.mini_batch_size

        if t.ndim != 1: t = np.argmax(t, axis=1)

        acc = 0.0

        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i * batch_size:(i + 1) * batch_size]
            tt = t[i * batch_size:(i + 1) * batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        return acc / x.shape[0]

    def gradient(self, x, t):
        loss = self.loss(x, t)

        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}

        for i in range(len(self.conv_param_list)):
            grads['Conv_W_' + str(i+1)] = self.layers['Conv_' + str(i+1)].dW
            grads['Conv_b_' + str(i+1)] = self.layers['Conv_' + str(i+1)].db

            if self.network_config['Conv_param_' + str(i+1)]['qat_scheme'] == KMQATScheme.LossAwareCompensation:
                grads['Compensation_alpha_' + str(i+1)] = self.layers['Compensation_' + str(i+1)].d_alpha
                grads['Compensation_mu_' + str(i+1)] = self.layers['Compensation_' + str(i+1)].d_mu
                grads['Compensation_offset_' + str(i+1)] = self.layers['Compensation_' + str(i+1)].d_offset

                if self.enable_compensation_L2_regularization:
                    grads['Compensation_mu_' + str(i + 1)] = self.layers['Compensation_' + str(i+1)].d_mu + \
                                                             self.compensation_L2_regularization_lambda * self.params['Compensation_mu_' + str(i+1)]
                    grads['Compensation_offset_' + str(i + 1)] = self.layers['Compensation_' + str(i + 1)].d_offset + \
                                                                 self.compensation_L2_regularization_lambda * self.params['Compensation_offset_' + str(i+1)]

                grads['BatchNorm_gamma_' + str(i+1)] = self.layers['BatchNorm_' + str(i+1)].dgamma
                grads['BatchNorm_beta_' + str(i+1)] = self.layers['BatchNorm_' + str(i+1)].dbeta

        for i in range(len(self.affine_param_list)):
            grads['Affine_W_' + str(i+1)] = self.layers['Affine_' + str(i+1)].dW
            grads['Affine_b_' + str(i+1)] = self.layers['Affine_' + str(i+1)].db

        return grads, loss