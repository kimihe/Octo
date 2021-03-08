# coding: utf-8
import sys, os
sys.path.append(os.pardir)
from common.optimizer import *

class Trainer:
    def __init__(self, network, x_train, t_train, x_test, t_test,
                 epochs=20, mini_batch_size=100,
                 optimizer='SGD', optimizer_param={'lr':0.01}, 
                 evaluate_sample_num_per_epoch=None, log_per_epoch=10, verbose=True):
        self.network = network
        self.verbose = verbose
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.epochs = epochs
        self.batch_size = mini_batch_size
        self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch

        optimizer_class_dict = {'sgd':SGD, 'momentum':Momentum, 'nesterov':Nesterov,
                                'adagrad':AdaGrad, 'rmsprpo':RMSprop, 'adam':Adam}
        self.optimizer = optimizer_class_dict[optimizer.lower()](**optimizer_param)
        
        self.train_size = x_train.shape[0]
        self.iter_per_epoch = max(self.train_size / mini_batch_size, 1)
        self.max_iter = int(epochs * self.iter_per_epoch)
        self.current_iter = 1
        self.current_epoch = 1
        self.log_interval = max(int(self.iter_per_epoch/log_per_epoch), 1)
        
        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

    def train_step(self):
        batch_mask = np.random.choice(self.train_size, self.batch_size)
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]
        
        grads, loss = self.network.gradient(x_batch, t_batch)
        self.optimizer.update(self.network.params, grads)

        if self.verbose:
            current_epoch_iter = self.current_iter % self.iter_per_epoch
            if current_epoch_iter == 0:
                current_epoch_iter = current_epoch_iter + self.iter_per_epoch
            print("Epoch: {}, \t iteration: [{}/{} ({:.2f}%)], \t train loss: {}".format(self.current_epoch,
                                                                                         current_epoch_iter,
                                                                                         self.iter_per_epoch,
                                                                                         current_epoch_iter / self.iter_per_epoch * 100,
                                                                                         loss))

        if self.current_iter % self.log_interval == 0:
            x_train_sample, t_train_sample = self.x_train, self.t_train
            x_test_sample, t_test_sample = self.x_test, self.t_test
            if not self.evaluate_sample_num_per_epoch is None:
                t = self.evaluate_sample_num_per_epoch
                x_train_sample, t_train_sample = self.x_train[:t], self.t_train[:t]
                x_test_sample, t_test_sample = self.x_test[:t], self.t_test[:t]

            train_acc = self.network.accuracy(x_train_sample, t_train_sample)
            test_acc = self.network.accuracy(x_test_sample, t_test_sample)
            self.train_acc_list.append(train_acc)
            self.test_acc_list.append(test_acc)
            self.train_loss_list.append(loss)

            if self.current_iter % self.iter_per_epoch == 0:
                if self.verbose:
                    print("============ Finished Epoch: {}, train acc: {}, test acc: {} ============".format(self.current_epoch, train_acc, test_acc))
                self.current_epoch += 1

        self.current_iter += 1

    def train(self, **kwargs):
        for i in range(self.max_iter):
            self.train_step()
        test_acc = self.network.accuracy(self.x_test, self.t_test)

        if self.verbose:
            print("============ Finished Training, Final Test Accuracy: {} ============ \n".format(test_acc))

