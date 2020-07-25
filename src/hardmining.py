import numpy as np
import torch
from torch import nn
from torch.utils.data.sampler import Sampler

from src.utils.util import find_device


def prepare_cifar_class(ds, threshold):
    return ds.data[np.array(ds.targets) == threshold]


def get_batch_random(batch_size, X):
    """
    Create batch of quadruplets with a complete random strategy

    Arguments:
      batch_size -- integer
      X          -- list containing n tensors of shape (?,w,h,c) to draw the batch from
    Returns:
      quadruplets -- list containing 4 tensors A,P,N,N2 of shape (batch_size,w,h,c)
    """
    n = max(X.targets)
    m, w, h, c = X.data.shape

    # initialize result
    quadruplets = [np.zeros((batch_size, h, w, c)) for i in range(4)]

    for i in range(batch_size):
        # pick one random class for anchor
        anchor_class = np.random.randint(0, n)
        anchor_class_examples = prepare_cifar_class(X, anchor_class)
        nb_sample_available_for_class_AP = len(anchor_class_examples)

        # pick two different examples - Anchor ; Anchor Positive
        [idx_A, idx_P] = np.random.choice(nb_sample_available_for_class_AP, size=2, replace=False)

        # pick other random class and example for Anchor Negative
        negative_class = (anchor_class + np.random.randint(1, n)) % n
        negative_class_examples = prepare_cifar_class(X, negative_class)
        nb_sample_available_for_class_N = len(negative_class_examples)

        idx_N = np.random.randint(0, nb_sample_available_for_class_N)

        # pick third class for second negative different from previous two
        remainingClasses = np.arange(n)
        np.delete(remainingClasses, [anchor_class, negative_class], axis=None)
        negative2_class = np.random.choice(remainingClasses, 1)[0]
        negative2_class_examples = prepare_cifar_class(X, negative2_class)
        nb_sample_available_for_class_N2 = len(negative2_class_examples)

        # example for Anchor Negative 2
        idx_N2 = np.random.randint(0, nb_sample_available_for_class_N2)

        quadruplets[0][i, :, :, :] = anchor_class_examples[idx_A]
        quadruplets[1][i, :, :, :] = anchor_class_examples[idx_P]
        quadruplets[2][i, :, :, :] = negative_class_examples[idx_N]
        quadruplets[3][i, :, :, :] = negative2_class_examples[idx_N2]

    return quadruplets


def get_batch_hard(draw_batch_size, hard_batchs_size, norm_batchs_size, network4, X):
    """
    Create batch of "hard" quadruplets

    Arguments:
      draw_batch_size -- number of initial randomly taken samples
      hard_batchs_size -- number of selected hardest samples to keep
      norm_batchs_size -- number of random samples to add
      network4 -- embedding module
      X -- dataset from which we select batch
    Returns:
      quadruplets -- list containing 4 tensors A,P,N,N2 of shape (hard_batchs_size+norm_batchs_size,w,h,c)
    """

    # pick a random batch to study
    studybatch = get_batch_random(draw_batch_size, X)

    # embeddings for each of anchors
    studybatch_ = torch.tensor(np.concatenate(studybatch, axis=0)).permute(0, 3, 1, 2).to(device=find_device(),
                                                                                          dtype=torch.float)

    network4.eval()
    APNN2 = network4(studybatch_)
    network4.train()

    tmp = torch.stack(torch.split(APNN2, 4))
    A = tmp[:, 0, ...]
    P = tmp[:, 1, ...]
    N = tmp[:, 2, ...]
    N2 = tmp[:, 3, ...]

    # compute the distances AP, AN, NN
    pairdist = nn.PairwiseDistance(2)  # euclidian distance between embeddings

    ap_dist = pairdist(A, P)
    an_dist = pairdist(A, N)
    nn_dist = pairdist(N, N2)

    # compute d(A,P) - d(A,N) + d(A,P) - d(N,N2)
    studybatchquadrupletloss = 2 * ap_dist - an_dist - nn_dist

    # Sort by distance
    selectionquadruplet = torch.argsort(studybatchquadrupletloss, descending=True)[:hard_batchs_size]

    # Draw other random samples from the batch
    selection2quadruplet = np.random.choice(np.delete(torch.arange(draw_batch_size), selectionquadruplet.cpu().data),
                                            norm_batchs_size, replace=False)
    selectionquadruplet = np.append(selectionquadruplet.cpu().data, selection2quadruplet)

    # batch of hard and regular samples - size (hard_batchs_size + norm_batchs_size)
    quadruplets = [studybatch[0][selectionquadruplet, :, :, :], studybatch[1][selectionquadruplet, :, :, :],
                   studybatch[2][selectionquadruplet, :, :, :], studybatch[3][selectionquadruplet, :, :, :]]

    return quadruplets


class CustomBatchSampler(Sampler):

    def __init__(self, data_source, mOHNM, batch_size=32, hard_batchs_size=32, norm_batchs_size=32,
                 embedding_network=None):
        super().__init__(data_source)
        self.dataset = data_source
        self.n_dataset = len(data_source)
        self.batch_size = batch_size
        self.hard_batchs_size = hard_batchs_size
        self.norm_batchs_size = norm_batchs_size
        self.mOHNM = mOHNM
        self.embedding = embedding_network

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            batch = []
            # hard minining strategy
            if self.mOHNM:
                batch.extend(
                    get_batch_hard(self.batch_size, self.hard_batchs_size, self.norm_batchs_size, self.embedding,
                                   self.dataset))
            # fully random batches of quadruples
            else:
                batch.extend(get_batch_random(self.batch_size, self.dataset))
            yield batch
            self.count += self.batch_size

    def __len__(self):
        return self.n_dataset // self.batch_size
