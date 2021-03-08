# coding: utf-8
import sys, os, datetime
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from simplecnn_lac.simplecnn_LAC_mnist import SimpleConvNet
from common.trainer import Trainer
from quantizer.KMQATScheme import KMQATScheme

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

train_data_num = 600  # max: 60000 of mnist
test_data_num = 100   # mnist: 1/6 of train_data_num
evaluate_sample_num = 100  # same as test_data_num
x_train, t_train = x_train[:train_data_num], t_train[:train_data_num]  # do not change
x_test, t_test = x_test[:test_data_num], t_test[:test_data_num]  # do not change

training_experiment_name = 'simplecnn_lac_mnist'  # do not change
training_experiment_date = datetime.datetime.now()  # do not change
max_epochs = 10  # max: 200
batch_size = 50  # do not larger than 100
learning_rate = 0.001  # do not change
optimizer = 'Adam'  # do not change
log_per_epoch = 10  # do not change
method_time_inspection = {}  # do not change

network = SimpleConvNet(input_dim=(1,28,28), 
                        conv_param_1={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1,
                                    'enable_fp_qat': True, 'enable_bp_gradient_quantization': True,
                                    'qat_scheme': KMQATScheme.LossAwareCompensation},
                        affine1_param={'enable_fp_qat': False, 'enable_bp_gradient_quantization': False},
                        affine2_param={'enable_fp_qat': False, 'enable_bp_gradient_quantization': False},
                        hidden_size=100, output_size=10, weight_init_std=0.01,
                        enable_compensation_L2_regularization=True, compensation_L2_regularization_lambda=0.1,
                        mini_batch_size=batch_size)

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
