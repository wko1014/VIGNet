from scipy.io import loadmat

import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

class load_dataset():
    def __init__(self, trial, cv, type="de_LDS", reg_label=False, call_eeg=False):
        self.trial = trial
        self.cv = cv
        self.type = type
        self.reg_label = reg_label
        self.call_eeg = call_eeg

        self.basePath = "/Define/your/own/path"

    def call(self):
        EEG = loadmat(self.basePath + "/Raw_Data/{}.mat".format(self.trial))["EEG"]["data"][0][0]
        feature = loadmat(self.basePath + "/DE/{}.mat".format(self.trial))[self.type]
        label = np.squeeze(loadmat(self.basePath + "/perclos_labels/{}.mat".format(self.trial))["perclos"])

        temp = np.zeros((feature.shape[1], int(EEG.shape[0] / feature.shape[1]), EEG.shape[-1]))  # (885, 1600, 17)
        for i in range(temp.shape[0]):
            temp[i, :, :] = EEG[i * 1600:(i + 1) * 1600, :]

        EEG = temp  # (885, 1600, 17)

        temp = np.zeros(shape=label.shape, dtype=np.int)

        for i in range(temp.shape[0]):
            if label[i] < 0.35:
                temp[i] = 0  # awake
            elif 0.35 <= label[i] < 0.7:
                temp[i] = 1  # tired
            else:
                temp[i] = 2  # drowsy

        clfLabel = np.eye(3)[temp]  # one-hot encoding / (855,3)
        feature = np.moveaxis(feature, 0, 1)  # feature.shape = (885, 17, 25)

        EEG = np.moveaxis(EEG, -1, 1)  # (885, 17, 1600)

        # We used five fold cross validation
        allIdx = np.random.RandomState(seed=970304).permutation(feature.shape[0])
        amount = int(feature.shape[0] / 5)

        testIdx = allIdx[self.cv * amount:(self.cv + 1) * amount]
        trainIdx = np.setdiff1d(allIdx, testIdx)

        amount = int(trainIdx.shape[0] / 5)
        randIdx = np.random.RandomState(seed=970304 + self.cv).permutation(trainIdx.shape[0])

        validIdx = trainIdx[randIdx[:amount]]
        trainIdx = np.setdiff1d(trainIdx, validIdx)

        trainFeature, validFeature, testFeature \
            = feature[trainIdx, :, :], feature[validIdx, :, :], feature[testIdx, :, :]
        trainEEG, validEEG, testEEG = EEG[trainIdx, :, :], EEG[validIdx, :, :], EEG[testIdx, :, :]
        trainLabel, validLabel, testLabel = clfLabel[trainIdx], clfLabel[validIdx], clfLabel[testIdx]
        trainReglabel, validReglabel, testReglabel = label[trainIdx], label[validIdx], label[testIdx]

        if self.call_eeg == True:
            trainFeature, validFeature, testFeature = trainEEG, validEEG, testEEG

        if self.reg_label == True:
            trainLabel, validLabel, testLabel = trainReglabel, validReglabel, testReglabel
            trainLabel, validLabel, testLabel \
                = np.expand_dims(trainLabel, -1), np.expand_dims(validLabel,-1), np.expand_dims(testLabel, -1)

        trainFeature, validFeature, testFeature \
            = np.expand_dims(trainFeature, -1), np.expand_dims(validFeature, -1), np.expand_dims(testFeature, -1)

        return trainFeature, trainLabel, validFeature, validLabel, testFeature, testLabel

def classification_loss(y, y_pred):
    return tf.keras.losses.binary_crossentropy(y, y_pred)

def regression_loss(y, y_pred):
    return tf.keras.losses.MSE(y, y_pred)

def grad(model, inputs, labels, mode):
    with tf.GradientTape() as tape:
        y_hat = model(inputs)

        if mode == "CLF":
            loss = classification_loss(y=labels, y_pred=y_hat)
        elif mode == "RGS":
            loss = classification_loss(y=labels, y_pred=y_hat)

    grad = tape.gradient(loss, model.trainable_variables)
    return loss, grad


class interpretative_feedback():
    def __init__(self, cv_idx, trial_idx):
        self.cv_idx = cv_idx
        self.trial_idx = trial_idx

        load_data = load_dataset(trial=self.trial_idx, cv=self.cv_idx, reg_label=False)
        _, self.train_label, _, _, _, _ = load_data.call()

        self.relevance = np.load("./relevances/cv{}_trial{}.npy".format(self.cv_idx, self.trial_idx))

    def plotting_relevance_map(self):
        R = self.relevance

        normal_idx = np.argmax(self.train_label, axis=-1) == 0 # 77
        tired_idx = np.argmax(self.train_label, axis=-1) == 1 # 233
        drowsy_idx = np.argmax(self.train_label, axis=-1) == 2 # 257

        R_normal, R_tired, R_drowsy = R[normal_idx, :, :, :], R[tired_idx, :, :, :], R[drowsy_idx, :, :, :]

        f, (ax1, ax2, ax3, axcb) = plt.subplots(4, 1, gridspec_kw={'height_ratios': [1, 1, 1, 0.08]})
        ax_normal = sns.heatmap(R_normal[56, :, :, 18], cbar=False, ax=ax1)
        ax_tired = sns.heatmap(R_tired[0, :, :, 0], cbar=False, ax=ax2)
        ax_drowsy = sns.heatmap(R_drowsy[0, :, :, 0], cbar_kws={"orientation":"horizontal"}, cbar_ax=axcb, ax=ax3)
        plt.show()
        plt.close()
