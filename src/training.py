import time

import numpy as np
import torch
from torch import nn

from src.quadloss import QuadrupletsLoss
from src.utils.evaluation import evaluate


def train_single_epoch(logger, opt, model, dataloader, device):
    dataloader.embedding = model
    criterion = QuadrupletsLoss()
    pairdist = nn.PairwiseDistance(2)  # euclidian distance between embeddings

    model.train()
    running_loss = 0.0
    start = time.time()
    for i, data_ in enumerate(dataloader):
        opt.zero_grad()
        # embeddings for each of anchors
        studybatch_ = torch.tensor(np.concatenate(data_, axis=0)).permute(0, 3, 1, 2).to(device=device,
                                                                                         dtype=torch.float)

        APNN2 = model(studybatch_)

        tmp = torch.stack(torch.split(APNN2, 4))
        A = tmp[:, 0, ...]
        P = tmp[:, 1, ...]
        N = tmp[:, 2, ...]
        N2 = tmp[:, 3, ...]

        # compute the distances AP, AN, NN

        output1 = pairdist(A, P)
        output2 = pairdist(A, N)
        output3 = pairdist(N, N2)

        distances = torch.stack([output1, output2, output3])
        loss = criterion(distances)
        loss.backward()
        opt.step()

        running_loss += loss.item()

        # log statistics
        if i % 10 == 9:
            logger.info('[%5d] loss: %.5f' % (i + 1, running_loss / 10))
            running_loss = 0.0

    end = time.time()
    logger.info(('Epoch took: %.5f minutes' % ((end - start) / 60)))

    return model


def train(logger, epochs, opt, model, dataloader, device, ds_train, ds_val):
    logger.info("=========== Let the training begin ===========")
    model = model.to(device)
    accuracies_train = []
    accuracies_val = []
    for e in range(epochs):
        logger.info("================== Epoch: %s ==================", e)
        train_single_epoch(logger, opt, model, dataloader, device)
        logger.info("evaluating ds_train")
        accuracies_train.append(evaluate(logger, ds_train, model, device))
        logger.info("evaluating ds_val")
        accuracies_val.append(evaluate(logger, ds_val, model, device))

    return accuracies_train, accuracies_val
