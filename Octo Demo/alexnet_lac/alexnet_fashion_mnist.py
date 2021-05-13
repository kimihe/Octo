# coding: utf-8
import sys, os
sys.path.append(os.pardir)  
import pickle
import numpy as np
from collections import OrderedDict
from common.layers import *

__DEBUG__ = False
__LOGTIME__ = True

class Alexnet:
    """AlexNet, 5 CONVs, 3 Affines

        conv1   - (compensation1 - batchnorm1) - relu_of_conv_1 - pool1 (max pool)
        conv2   - (compensation2 - batchnorm2) - relu_of_conv_2 - pool2 (max pool)
        conv3   - (compensation3 - batchnorm3) - relu_of_conv_3
        conv4   - (compensation4 - batchnorm4) - relu_of_conv_4
        conv5   - (compensation5 - batchnorm5) - relu_of_conv_5 - pool3 (max pool)
        affine1 - relu_of_affine_1 - dropout1
        affine2 - relu_of_affine_2 - dropout2
        affine3
        softmax
    """

    def __init__(self, input_dim=(1, 28, 28),  # Fashion MNIST
                 conv_param_1={'filter_num': 96, 'filter_size': 11, 'pad': 1, 'stride': 1,
                               'enable_fp_qat': False, 'enable_bp_gradient_quantization': False,
                               'qat_scheme': KMQATScheme.FullPrecision},
                 conv_param_2={'filter_num': 256, 'filter_size': 5, 'pad': 1, 'stride': 1,
                               'enable_fp_qat': False, 'enable_bp_gradient_quantization': False,
                               'qat_scheme': KMQATScheme.FullPrecision},
                 conv_param_3={'filter_num': 384, 'filter_size': 3, 'pad': 1, 'stride': 1,
                               'enable_fp_qat': False, 'enable_bp_gradient_quantization': False,
                               'qat_scheme': KMQATScheme.FullPrecision},
                 conv_param_4={'filter_num': 384, 'filter_size': 3, 'pad': 1, 'stride': 1,
                               'enable_fp_qat': False, 'enable_bp_gradient_quantization': False,
                               'qat_scheme': KMQATScheme.FullPrecision},
                 conv_param_5={'filter_num': 256, 'filter_size': 3, 'pad': 1, 'stride': 1,
                               'enable_fp_qat': False, 'enable_bp_gradient_quantization': False,
                               'qat_scheme': KMQATScheme.FullPrecision},
                 affine1_param={'enable_fp_qat': False, 'enable_bp_gradient_quantization': False},
                 affine2_param={'enable_fp_qat': False, 'enable_bp_gradient_quantization': False},
                 affine3_param={'enable_fp_qat': False, 'enable_bp_gradient_quantization': False},
                 hidden_size_1=4096, hidden_size_2=4096, output_size=10,
                 enable_compensation_L2_regularization=False, compensation_L2_regularization_lambda=0.1,
                 mini_batch_size=100):

        # 0. Log Time =================================
        if __LOGTIME__:
            self.method_time_inspection = {}
            self.fp_time_list = []
            self.bp_time_list = []
            self.per_iteration_time_list = []
            self.method_time_inspection['FORWARD_PASS'] = self.fp_time_list
            self.method_time_inspection['BACKWARD_PASS'] = self.bp_time_list
            self.method_time_inspection['PER_ITERATION'] = self.per_iteration_time_list

        # 1. Basic setting =================================
        self.network_name = type(self).__name__
        self.conv_param_list = [conv_param_1, conv_param_2, conv_param_3, conv_param_4, conv_param_5]
        self.affine_param_list = [affine1_param, affine2_param, affine3_param]
        self.enable_compensation_L2_regularization = enable_compensation_L2_regularization
        self.compensation_L2_regularization_lambda = compensation_L2_regularization_lambda
        self.mini_batch_size = mini_batch_size

        # Check invalid configuration
        for i, v in enumerate(self.conv_param_list):
            if v['qat_scheme'] == KMQATScheme.FullPrecision:
                v['enable_fp_qat'] = False
                v['enable_bp_gradient_quantization'] = False

        self.network_config = {}
        for i, v in enumerate(self.conv_param_list):
            self.network_config['Conv_param_' + str(i + 1)] = v

        for i, v in enumerate(self.affine_param_list):
            self.network_config['Affine_param_' + str(i + 1)] = v

        # 2. Init neuron numbers =================================
        pre_node_nums = np.ones(len(self.conv_param_list) + len(self.affine_param_list), dtype=np.int)
        # Conv layers
        pre_channel_num = input_dim[0]
        for i in range(len(self.conv_param_list)):
            pre_node_nums[i] = pre_channel_num * (self.network_config['Conv_param_' + str(i+1)]['filter_size'] ** 2)
            pre_channel_num = self.network_config['Conv_param_' + str(i+1)]['filter_num']
        # Affine layers
        pre_node_nums[5] = pre_channel_num * 2 * 2  # pool_output_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2))
        pre_node_nums[6] = hidden_size_1
        pre_node_nums[7] = hidden_size_2
        wight_init_scales = np.sqrt(2.0 / pre_node_nums)  

        # 3. Init parameters of Convs and Affines =================================
        self.params = {}
        pre_channel_num = input_dim[0]
        for i in range(len(self.conv_param_list)):
            # W shape: (output_channel, input_channel, filter_height, filter_width)
            self.params['Conv_W_' + str(i+1)] = wight_init_scales[i] * np.random.randn(self.network_config['Conv_param_' + str(i+1)]['filter_num'],
                                                                                       pre_channel_num,
                                                                                       self.network_config['Conv_param_' + str(i+1)]['filter_size'],
                                                                                       self.network_config['Conv_param_' + str(i+1)]['filter_size'])
            # b shape: (output_channel, 1, 1), b will be broadcasted to W
            self.params['Conv_b_' + str(i+1)] = np.zeros(self.network_config['Conv_param_' + str(i+1)]['filter_num'])
            pre_channel_num = self.network_config['Conv_param_' + str(i+1)]['filter_num']

        self.params['Affine_W_1'] = wight_init_scales[5] * np.random.randn(pre_channel_num * 2 * 2, hidden_size_1)
        self.params['Affine_b_1'] = np.zeros(hidden_size_1)
        self.params['Affine_W_2'] = wight_init_scales[6] * np.random.randn(hidden_size_1, hidden_size_2)
        self.params['Affine_b_2'] = np.zeros(hidden_size_2)
        self.params['Affine_W_3'] = wight_init_scales[7] * np.random.randn(hidden_size_2, output_size)
        self.params['Affine_b_3'] = np.zeros(output_size)

        # 4. Init parameters of Compensation and BatchNorm layers =================================
        compensation_param_shape_list = [(self.mini_batch_size, 96, 20, 20), (self.mini_batch_size, 256, 8, 8),
                                         (self.mini_batch_size, 384, 4, 4), (self.mini_batch_size, 384, 4, 4),
                                         (self.mini_batch_size, 256, 4, 4)]
        # batchnorm_param_shape_list = [(100, 38400), (100, 16384), (100, 6144), (100, 6144), (100, 4096)]
        batchnorm_param_shape_list = []
        for i in range(len(compensation_param_shape_list)):
            ele = (compensation_param_shape_list[i][0],
                   compensation_param_shape_list[i][1] * compensation_param_shape_list[i][2] * compensation_param_shape_list[i][3])
            batchnorm_param_shape_list.append(ele)

        # 5. Build neuron network =================================
        # Network construction via ordered dictionary, 5 CONVs, 3 FCs
        self.layers = OrderedDict()
        for i in range(len(self.conv_param_list)):
            # Convolutional Layers
            self.layers['Conv_' + str(i+1)] = Convolution(self.params['Conv_W_' + str(i+1)],
                                                          self.params['Conv_b_' + str(i+1)],
                                                          self.network_config['Conv_param_' + str(i+1)]['stride'],
                                                          self.network_config['Conv_param_' + str(i+1)]['pad'],
                                                          enable_fp_qat=self.network_config['Conv_param_' + str(i+1)]['enable_fp_qat'],
                                                          enable_bp_gradient_quantization=self.network_config['Conv_param_' + str(i+1)]['enable_bp_gradient_quantization'],
                                                          layer_id='Conv_' + str(i+1),
                                                          qat_scheme=self.network_config['Conv_param_' + str(i+1)]['qat_scheme'])

            # Compensation Layers and Batch Normalization
            if self.network_config['Conv_param_' + str(i+1)]['qat_scheme'] == KMQATScheme.LossAwareCompensation:
                self.params['Compensation_alpha_' + str(i+1)] = np.ones(compensation_param_shape_list[i], dtype=np.float64)
                self.params['Compensation_mu_' + str(i+1)] = 1e-3 * np.random.normal(0.0, 1.0, size=compensation_param_shape_list[i])
                self.params['Compensation_offset_' + str(i+1)] = 1e-3 * np.random.normal(0.0, 1.0, size=compensation_param_shape_list[i])
                self.layers['Compensation_' + str(i+1)] = Compensation(self.params['Compensation_alpha_' + str(i+1)],
                                                                       self.params['Compensation_mu_' + str(i+1)],
                                                                       self.params['Compensation_offset_' + str(i+1)],
                                                                       layer_id='Compensation_' + str(i+1))

                self.params['BatchNorm_gamma_' + str(i+1)] = np.ones(batchnorm_param_shape_list[i])  # (batch_size, filter_num * width * height)
                self.params['BatchNorm_beta_' + str(i+1)] = np.zeros(batchnorm_param_shape_list[i])
                self.layers['BatchNorm_' + str(i+1)] = BatchNormalization(self.params['BatchNorm_gamma_' + str(i+1)],
                                                                          self.params['BatchNorm_beta_' + str(i+1)],
                                                                          layer_id='BatchNorm_' + str(i+1))

            # Activation: ReLU
            self.layers['Relu_of_Conv_' + str(i+1)] = Relu()

            # Pooling
            if i == 0 or i == 1 or i == 4:
                self.layers['Pool_' + str(i+1)] = Pooling(pool_h=2, pool_w=2, stride=2)

        for i in range(len(self.affine_param_list)):
            # Affine Layers: Fully Connected Layers
            self.layers['Affine_' + str(i+1)] = Affine(self.params['Affine_W_' + str(i+1)],
                                                       self.params['Affine_b_' + str(i+1)],
                                                       enable_fp_qat=self.network_config['Affine_param_' + str(i+1)]['enable_fp_qat'],
                                                       enable_bp_gradient_quantization=self.network_config['Affine_param_' + str(i+1)]['enable_bp_gradient_quantization'],
                                                       layer_id='Affine_' + str(i+1))

            # Activation: ReLU
            if i == 0 or i == 1:
                self.layers['Relu_of_Affine_' + str(i+1)] = Relu()

            # Dropout Layers
            if i == 0 or i == 1:
                self.layers['Dropout_' + str(i+1)] = Dropout(0.5)

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x, train_flg=False):
        # for layer in self.layers:
        for layer in self.layers.values():
            if isinstance(layer, Dropout):
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x

    def loss(self, x, t):
        __DEBUG__ = False
        if __DEBUG__:
            print("self.layers['Conv_3'].W.max: {}".format(np.max(self.layers['Conv_3'].W)))
            print("self.params['Conv_W_3'].max: {} ".format(np.max(self.params['Conv_W_3'])))
            print("Two tensors are equal: {} \n".format(np.array_equal(self.layers['Conv_3'].W, self.params['Conv_W_3'])))

            print("self.layers['Conv_3'].b.max: {}".format(np.max(self.layers['Conv_3'].b)))
            print("self.params['Conv_b_3'].max: {}".format(np.max(self.params['Conv_b_3'])))
            print("Two tensors are equal: {} \n".format(np.array_equal(self.layers['Conv_3'].b, self.params['Conv_b_3'])))

            print("self.layers['Compensation_3'].alpha.max: {}".format(np.max(self.layers['Compensation_3'].alpha)))
            print("self.params['Compensation_alpha_3'].max: {}".format(np.max(self.params['Compensation_alpha_3'])))
            print("Two tensors are equal: {} \n".format(np.array_equal(self.layers['Compensation_3'].alpha, self.params['Compensation_alpha_3'])))

            print("self.layers['Compensation_3'].mu.max: {}".format(np.max(self.layers['Compensation_3'].mu)))
            print("self.params['Compensation_mu_3'].max: {}".format(np.max(self.params['Compensation_mu_3'])))
            print("Two tensors are equal: {} \n".format(np.array_equal(self.layers['Compensation_3'].mu, self.params['Compensation_mu_3'])))

            print("self.layers['Compensation_3'].offset.max: {}".format(np.max(self.layers['Compensation_3'].offset)))
            print("self.params['Compensation_offset_3'].max: {}".format(np.max(self.params['Compensation_offset_3'])))
            print("Two tensors are equal: {} \n".format(np.array_equal(self.layers['Compensation_3'].offset, self.params['Compensation_offset_3'])))

            print("self.layers['BatchNorm_3'].gamma.max: {}".format(np.max(self.layers['BatchNorm_3'].gamma)))
            print("self.params['BatchNorm_gamma_3'].max: {}".format(np.max(self.params['BatchNorm_gamma_3'])))
            print("Two tensors are equal: {} \n".format(np.array_equal(self.layers['BatchNorm_3'].gamma, self.params['BatchNorm_gamma_3'])))

            print("self.layers['BatchNorm_3'].beta.max: {}".format(np.max(self.layers['BatchNorm_3'].beta)))
            print("self.params['BatchNorm_beta_3'].max: {}".format(np.max(self.params['BatchNorm_beta_3'])))
            print("Two tensors are equal: {} \n".format(np.array_equal(self.layers['BatchNorm_3'].beta, self.params['BatchNorm_beta_3'])))

            print("self.layers['Affine_1'].W.max: {}".format(np.max(self.layers['Affine_1'].W)))
            print("self.params['Affine_W_1'].max: {}".format(np.max(self.params['Affine_W_1'])))
            print("Two tensors are equal: {} \n".format(np.array_equal(self.layers['Affine_1'].W, self.params['Affine_W_1'])))

            print("self.layers['Affine_1'].b.max: {}".format(np.max(self.layers['Affine_1'].b)))
            print("self.params['Affine_b_1'].max: {}".format(np.max(self.params['Affine_b_1'])))
            print("Two tensors are equal: {} \n".format(np.array_equal(self.layers['Affine_1'].b, self.params['Affine_b_1'])))
            print("===================================")

        if self.enable_compensation_L2_regularization:
            y = self.predict(x, train_flg=True)

            # L2 Regularization for Loss-aware Compensation
            l2_regularization = 0
            for i in range(len(self.conv_param_list)):
                if self.network_config['Conv_param_' + str(i+1)]['qat_scheme'] == KMQATScheme.LossAwareCompensation:
                    mu = self.params['Compensation_mu_' + str(i+1)]
                    offset = self.params['Compensation_offset_' + str(i+1)]
                    l2_regularization += 0.5 * self.compensation_L2_regularization_lambda * (np.sum(mu ** 2) + np.sum(offset ** 2))
            return self.last_layer.forward(y, t) + l2_regularization
        else:
            y = self.predict(x, train_flg=True)
            return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        batch_size = self.mini_batch_size

        if t.ndim != 1: t = np.argmax(t, axis=1)

        acc = 0.0

        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i * batch_size:(i + 1) * batch_size]
            tt = t[i * batch_size:(i + 1) * batch_size]
            y = self.predict(tx, train_flg=False)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        return acc / x.shape[0]

    def gradient(self, x, t):
        if __LOGTIME__:
            start = time.time()

        # forward
        loss = self.loss(x, t)

        if __LOGTIME__:
            end_fp = time.time()
            # print("Forward Pass Time Cost (ms): {:.2f}".format((end_fp - start) * 1000))
            self.method_time_inspection['FORWARD_PASS'].append(str("{:.2f}".format((end_fp - start) * 1000)))

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        # tmp_layers = self.layers.copy()
        # tmp_layers.reverse()
        # for layer in tmp_layers:
        #     dout = layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        if __LOGTIME__:
            end_bp = time.time()
            # print("Backward Pass Time Cost (ms): {:.2f}".format((end_bp - end_fp) * 1000))
            self.method_time_inspection['BACKWARD_PASS'].append(str("{:.2f}".format((end_bp - end_fp) * 1000)))

        grads = {}

        for i in range(len(self.conv_param_list)):
            # Gradients of CONV's weight and bias
            grads['Conv_W_' + str(i+1)] = self.layers['Conv_' + str(i+1)].dW
            grads['Conv_b_' + str(i+1)] = self.layers['Conv_' + str(i+1)].db

            if self.network_config['Conv_param_' + str(i+1)]['qat_scheme'] == KMQATScheme.LossAwareCompensation:
                # Gradients of compensation layer
                grads['Compensation_alpha_' + str(i+1)] = self.layers['Compensation_' + str(i+1)].d_alpha
                grads['Compensation_mu_' + str(i+1)] = self.layers['Compensation_' + str(i+1)].d_mu
                grads['Compensation_offset_' + str(i+1)] = self.layers['Compensation_' + str(i+1)].d_offset

                if self.enable_compensation_L2_regularization:
                    grads['Compensation_mu_' + str(i + 1)] = self.layers['Compensation_' + str(i+1)].d_mu + \
                                                             self.compensation_L2_regularization_lambda * self.params['Compensation_mu_' + str(i+1)]
                    grads['Compensation_offset_' + str(i + 1)] = self.layers['Compensation_' + str(i + 1)].d_offset + \
                                                                 self.compensation_L2_regularization_lambda * self.params['Compensation_offset_' + str(i+1)]

                # gradients of Batch Normalization
                grads['BatchNorm_gamma_' + str(i+1)] = self.layers['BatchNorm_' + str(i+1)].dgamma
                grads['BatchNorm_beta_' + str(i+1)] = self.layers['BatchNorm_' + str(i+1)].dbeta

        for i in range(len(self.affine_param_list)):
            # Gradients of Affine's weight and bias
            grads['Affine_W_' + str(i+1)] = self.layers['Affine_' + str(i+1)].dW
            grads['Affine_b_' + str(i+1)] = self.layers['Affine_' + str(i+1)].db

        if __LOGTIME__:
            end_iteration = time.time()
            # print("Per Iteration Time Cost (ms): {:.2f}".format((end_iteration - start) * 1000))
            self.method_time_inspection['PER_ITERATION'].append(str("{:.2f}".format((end_iteration - start) * 1000)))

        return grads, loss

    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i in range(len(self.conv_param_list)):
            # CONV's weight and bias
            self.layers['Conv_' + str(i + 1)].W = self.params['Conv_W_' + str(i + 1)]
            self.layers['Conv_' + str(i + 1)].b = self.params['Conv_b_' + str(i + 1)]

        for i in range(len(self.conv_param_list)):
            # Three learnable parameters for Loss-aware Compensation
            self.layers['Compensation_' + str(i + 1)].alpha = self.params['Compensation_alpha_' + str(i + 1)]
            self.layers['Compensation_' + str(i + 1)].mu = self.params['Compensation_mu_' + str(i + 1)]
            self.layers['Compensation_' + str(i + 1)].offset = self.params['Compensation_offset_' + str(i + 1)]

        for i in range(len(self.affine_param_list)):
            # Affine's  weight and bias
            self.layers['Affine_' + str(i + 1)].W = self.params['Affine_W_' + str(i + 7)]
            self.layers['Affine_' + str(i + 1)].b = self.params['Affine_b_' + str(i + 7)]
