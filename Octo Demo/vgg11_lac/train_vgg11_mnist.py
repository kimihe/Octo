# coding: utf-8
import sys, os, datetime, resource, logging
sys.path.append(os.pardir) 
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
import pandas as pd
from dataset.mnist import load_mnist
from vgg11_lac.vgg11_mnist import VGG_11
from common.trainer import Trainer
from quantizer.KMQATScheme import KMQATScheme

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)  # MNIST

# ======================================= Training Configuration =======================================
train_data_num = 200  # max: 50000 of cifar10, 60000 of mnist
test_data_num = 100   # cifar10: 1/5 of train_data_num, mnist: 1/6 of train_data_num
evaluate_sample_num = 200  # cannot be too small, can be set with the same value as train_data_num
x_train, t_train = x_train[:train_data_num], t_train[:train_data_num]  # do not change
x_test, t_test = x_test[:test_data_num], t_test[:test_data_num]  # do not change

training_experiment_name = 'vgg11_lac_mnist'  # do not change
training_experiment_date = datetime.datetime.now()  # do not change
max_epochs = 2  # max: 200
batch_size = 50  # do not larger than 100
learning_rate = 0.001  # do not change
optimizer = 'Adam'  # do not change
log_per_epoch = 10  # do not change
method_time_inspection = {}  # do not change
# ======================================= Training Configuration =======================================

# ======================================= Network Configuration =======================================
# VGG11: 8 Convs + 3 Affines
network = VGG_11(input_dim=(1, 28, 28),
                 conv_param_1={'filter_num': 64, 'filter_size': 3, 'pad': 0, 'stride': 1,
                               'enable_fp_qat': True, 'enable_bp_gradient_quantization': True,
                               'qat_scheme': KMQATScheme.LossAwareCompensation},
                 conv_param_2={'filter_num': 128, 'filter_size': 3, 'pad': 0, 'stride': 1,
                               'enable_fp_qat': True, 'enable_bp_gradient_quantization': True,
                               'qat_scheme': KMQATScheme.LossAwareCompensation},
                 conv_param_3={'filter_num': 256, 'filter_size': 3, 'pad': 0, 'stride': 1,
                               'enable_fp_qat': True, 'enable_bp_gradient_quantization': True,
                               'qat_scheme': KMQATScheme.LossAwareCompensation},
                 conv_param_4={'filter_num': 256, 'filter_size': 3, 'pad': 0, 'stride': 1,
                               'enable_fp_qat': True, 'enable_bp_gradient_quantization': True,
                               'qat_scheme': KMQATScheme.LossAwareCompensation},
                 conv_param_5={'filter_num': 512, 'filter_size': 3, 'pad': 0, 'stride': 1,
                               'enable_fp_qat': True, 'enable_bp_gradient_quantization': True,
                               'qat_scheme': KMQATScheme.LossAwareCompensation},
                 conv_param_6={'filter_num': 512, 'filter_size': 3, 'pad': 0, 'stride': 1,
                               'enable_fp_qat': True, 'enable_bp_gradient_quantization': True,
                               'qat_scheme': KMQATScheme.LossAwareCompensation},
                 conv_param_7={'filter_num': 512, 'filter_size': 3, 'pad': 0, 'stride': 1,
                               'enable_fp_qat': True, 'enable_bp_gradient_quantization': True,
                               'qat_scheme': KMQATScheme.LossAwareCompensation},
                 conv_param_8={'filter_num': 512, 'filter_size': 3, 'pad': 0, 'stride': 1,
                               'enable_fp_qat': True, 'enable_bp_gradient_quantization': True,
                               'qat_scheme': KMQATScheme.LossAwareCompensation},
                 affine1_param={'enable_fp_qat': False, 'enable_bp_gradient_quantization': False},
                 affine2_param={'enable_fp_qat': False, 'enable_bp_gradient_quantization': False},
                 affine3_param={'enable_fp_qat': False, 'enable_bp_gradient_quantization': False},
                 hidden_size_1=4096, hidden_size_2=1000, output_size=10,
                 enable_compensation_L2_regularization=True, compensation_L2_regularization_lambda=0.1,
                 mini_batch_size=batch_size)
# ======================================= Network Configuration =======================================

trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=max_epochs, mini_batch_size=batch_size,
                  optimizer=optimizer, optimizer_param={'lr': learning_rate},
                  evaluate_sample_num_per_epoch=evaluate_sample_num, log_per_epoch=log_per_epoch, verbose=True)
trainer.train(log_time=method_time_inspection)

# Draw figure
markers = {'train loss': '^', 'train acc': 'o', 'test acc': 's'}
x = np.arange(len(trainer.train_loss_list))
plt.plot(x, trainer.train_loss_list, marker='^', label='train loss', markevery=2)
plt.plot(x, trainer.train_acc_list, marker='o', label='train acc', markevery=2)
plt.plot(x, trainer.test_acc_list, marker='s', label='test acc', markevery=2)
plt.xlabel("Training Progress")
plt.ylabel("Accuracy & Loss")
plt.ylim(0, 2.5)
plt.legend(loc='best')
plt.show()
