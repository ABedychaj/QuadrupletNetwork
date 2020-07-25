# QuadrupletNetwork
Simple Quadruplet Network implementation for image embeddings. 

Quadruplet network is loosely based on the paper ["Beyond triplet loss: a deep quadruplet network for person re-identification" by Weihua Chen et al.](https://arxiv.org/abs/1704.01719) [1]. 
The main difference is that it is adopted to CIFAR100 dataset, and metric net with softmax layer (from the paper) is ephemerally present in the form of nearest neighbor search with euclidean distance.
Online hard mining is reimplemented in Pytorch based on article: [Beyond triplet loss : One shot learning experiments with quadruplet loss](https://medium.com/@crimy/beyond-triplet-loss-one-shot-learning-experiments-with-quadruplet-loss-16671ed51290) [2] - where the original mining strategy is done in Keras.

###### Evaluation:
The metric that is used in evaluation is Accuracy@K, i.e. the proportion of times an image of the same class is found in the top K neighbours of a given image. 

To emulate a realistic scenario with test images of previously unseen classes, CIFAR100 dataset is adopted by limiting the training dataset to 50 classes, while using the remaining 50 classes for evaluation.

###### Dependencies:

`pip install -r requirement.txt`

###### References:
[1] Chen, W., Chen, X., Zhang, J., & Huang, K. (2017). Beyond Triplet Loss: A Deep Quadruplet Network for Person Re-identification. 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 1320-1329.

[2] <https://medium.com/@crimy/beyond-triplet-loss-one-shot-learning-experiments-with-quadruplet-loss-16671ed51290>
