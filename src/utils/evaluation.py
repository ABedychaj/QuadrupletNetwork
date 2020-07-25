import numpy as np
from sklearn.neighbors import KDTree
from torch.utils.data import DataLoader


def dump_embeddings(logger, dataloader, model, device):
    embeddings = []
    for i, batch in enumerate(dataloader):
        if i % 50 == 0:
            logger.debug("eval batch %s", i)
        embeddings_batch = model(batch[0].to(device))
        embeddings.append(embeddings_batch.cpu().detach().numpy())

    return np.vstack(embeddings)


def accuracy_at_k(y_true, embeddings, K, sample):
    kdtree = KDTree(embeddings)
    if sample is None:
        sample = len(y_true)
    y_true_sample = y_true[:sample]

    indices_of_neighbours = kdtree.query(embeddings[:sample], k=K + 1, return_distance=False)[:, 1:]

    y_hat = y_true[indices_of_neighbours]

    matching_category_mask = np.expand_dims(np.array(y_true_sample), -1) == y_hat

    matching_cnt = np.sum(matching_category_mask.sum(-1) > 0)
    accuracy = matching_cnt / len(y_true_sample)
    return accuracy


def evaluate(logger, dataset, embedding_model, device):
    embedding_model.eval()
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    embeddings = dump_embeddings(logger, dataloader, embedding_model, device)
    accuracies = []
    for K in [1, 5, 10]:
        acc_k = accuracy_at_k(np.array(dataset.targets), embeddings, K, 200)
        accuracies.append(acc_k)
        logger.info("accuracy@{} = {}".format(K, acc_k))
    return accuracies
