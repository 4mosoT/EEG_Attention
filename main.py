from typing_extensions import Required
from scipy import signal
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing, model_selection, metrics


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

import nets

from biosip_tools.eeg import utils
from biosip_tools.eeg import timeseries
# https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9669037

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="path to data")
    parser.add_argument('stim', type=str, help="stim frequency")
    parser.add_argument('-ws', '--window_size', type=int,
                        default=1, help="window size")
    parser.add_argument('-bs', '--batch_size', type=int,
                        default=1, help="batch size")
    parser.add_argument('-ep', '--epochs', type=int, default=1, help="epochs")

    args = parser.parse_args()

    data_path = args.path
    stim = args.stim

    eeg_dlx = timeseries.EEGSeries(path=data_path + f"EEG_7_{stim}_RD.npy")
    eeg_ctrl = timeseries.EEGSeries(path=data_path + f"EEG_7_{stim}_C.npy")
    eeg = eeg_ctrl.append(eeg_dlx)
    eeg.apply_cheby_filter(0.5, 80)
    eeg.data = preprocessing.StandardScaler().fit_transform(
        eeg.data.reshape(eeg.data.shape[1], -1)).reshape(eeg.data.shape)

    generator = utils.window_data_loader(eeg, window_size=args.window_size)
    batch, labels, _ = generator.__next__()
    _, EEGChannels, EEGSamples = batch.shape

    writer = SummaryWriter()
    kf = model_selection.StratifiedKFold(n_splits=5)
    # Ver si mejor dividir entre sujetos o a nivel de ventanas
    labels = np.append(
        np.ones(eeg_dlx.data.shape[0]), np.zeros(eeg_ctrl.data.shape[0]))

    weight = eeg_ctrl.data.shape[0] / eeg_dlx.data.shape[0]

    for fold, (train_index, test_index) in enumerate(kf.split(np.zeros(labels.shape), labels)):

        net = nets.CompleteNet(EEGChannels, EEGSamples).to("cuda")
        loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(
            [1, weight]).to("cuda"))  # Pos weight balance
        optimizer = Adam(net.parameters(), lr=0.001)

        trainEEG = timeseries.EEGSeries(data=eeg.data[train_index])
        train_labels = labels[train_index]

        testEEG = timeseries.EEGSeries(data=eeg.data[test_index])
        test_labels = labels[test_index]

        train_generator = utils.window_data_loader(
            trainEEG, window_size=args.window_size, batch_size=args.batch_size, labels=train_labels, epochs=args.epochs, shuffle=True)
        losses = []

        for step, (batch, batch_labels, epoch) in enumerate(train_generator):

            optimizer.zero_grad()
            batch_labels = np.vstack(
                (1 - batch_labels, batch_labels)).T  # DLX = 0, CTRL = 1

            #weight_vector = batch_labels[:, 1] * weight

            batch = torch.from_numpy(batch).type(torch.FloatTensor).unsqueeze(
                1).to("cuda")  # Unsqueeze to add channel dimension
            batch_loss = loss(net(batch)[0], torch.from_numpy(
                batch_labels).to("cuda"))
            batch_loss.backward()
            optimizer.step()

            if epoch is not None:
                with torch.no_grad():
                    test_results = []

                    test_generator = utils.window_data_loader(
                        testEEG, window_size=args.window_size, labels=test_labels, epochs=1, return_subjects=True)
                    for subject_batch, test_labels, _ in test_generator:
                        subject_batch = torch.from_numpy(subject_batch).type(
                            torch.FloatTensor).unsqueeze(1).to("cuda")

                        lbls = np.vstack((1 - test_labels, test_labels)).T
                        test_loss = loss(net(subject_batch)[0], torch.from_numpy(
                            lbls).to("cuda")).cpu().numpy()

                        result = torch.sigmoid(
                            net(subject_batch)[0]).cpu().numpy().mean(axis=0)
                        test_results.append((test_labels[0], np.argmax(
                            result, axis=0), np.mean(test_loss)))

                    test_results = np.array(test_results)
                    acc = metrics.accuracy_score(
                        test_results[:, 1], test_results[:, 0])

                    writer.add_scalars("Fold_{}".format(fold), {'train_loss': batch_loss.item(),
                                                                'test_loss': test_results[:, -1].mean(),
                                                                'accuracy': acc}, epoch)
