import utils
import network

import tensorflow as tf
import numpy as np
import os

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

class experiment():
    def __init__(self, trial_idx, cv_idx, gpu_idx, task):
        # # Assign GPU
        # tf.debugging.set_log_device_placement(True)
        # self.gpu_idx = gpu_idx
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_idx)
        # tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
        
        self.trial_idx = trial_idx
        self.cv_idx = cv_idx
        self.task = task # for controlling task of interests

        self.reg_label = False

        if self.task == "RGS":
            self.reg_label = True

        # Define learning schedules
        self.learning_rate = 1e-3
        self.num_epochs = 100
        self.num_batches = 5
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def training(self):
        print("START TRAINING TRIAL {} CV {}".format(self.trial_idx, self.cv_idx))
        # Load dataset
        load_data = utils.load_dataset(trial=self.trial_idx, cv=self.cv_idx, reg_label=self.reg_label)
        Xtrain, Ytrain, Xvalid, Yvalid, Xtest, Ytest = load_data.call()
        Yvalid, Ytest = np.argmax(Yvalid, axis=-1), np.argmax(Ytest, axis=-1)

        # Call model
        VIGNet = network.vignet(mode=self.task)

        # Optimization
        optimizer = self.optimizer
        num_batch_iter = int(Xtrain.shape[0]/self.num_batches)

        for epoch in range(self.num_epochs):
            loss_per_epoch = 0
            # Randomize the training dataset
            rand_idx = np.random.permutation(Xtrain.shape[0])
            Xtrain, Ytrain = Xtrain[rand_idx, :, :, :], Ytrain[rand_idx, :]

            for batch in range(num_batch_iter):
                # Sample minibatch
                x = Xtrain[batch * self.num_batches:(batch + 1) * self.num_batches, :, :, :]
                y = Ytrain[batch * self.num_batches:(batch + 1) * self.num_batches, :]

                # Estimate loss
                loss, grads = utils.grad(model=VIGNet, inputs=x, labels=y, mode=self.task)

                # Update the network
                optimizer.apply_gradients(zip(grads, VIGNet.trainable_variables))
                loss_per_epoch += np.mean(loss)

            print("Epoch: {}, Training Loss: {:0.4}".format(epoch + 1, loss_per_epoch/num_batch_iter))
        print("\n")
        
for trial in range(1, 24):
    for fold in range(5):
        main = experiment(trial_idx=trial, cv_idx=fold, gpu_idx=0, task='CLF') # task='RGS' for regression
        main.training()
