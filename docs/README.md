**IMPORTANT:** Please view documents on [Github Pages](https://ecr23xx.github.io/cs231n)

## Solutions

* [Assignment 1](https://github.com/ECer23/cs231n.assignments/tree/master/assignment1) Complete 100%
* [Assignment 2](https://github.com/ECer23/cs231n.assignments/tree/master/assignment2) Complete 85%
* [Assignment 3](https://github.com/ECer23/cs231n.assignments/tree/master/assignment3) Complete 0%

## Notes

* [1 - Classifier](1-Classifier) kNN, SVM, Softmax and basic Neural Network classifier.
* [2 - Convolutional Neural Network](2-Convolutional-Neural-Network) ConvNets architecture design, fine tuning tips and vectorized implementation details.
* [3 - ConvNets Training Tips](3-ConvNets-Training-Tips) Tips on setting up data, model and loss functions to boost your ConvNets training

## Motivation

When I'm doing another project that based on [yolov3.pytorch](https://github.com/ECer23/yolov3.pytorch), I came across a problem in Batch Normalization. My understanding of BN just limits in `nn.BatchNorm`, and when I want to do things like computing accumulated gradients, I got stuck in because I'm not familiar with computation of running mean/variance or something like that. This is the motivation of doing CS231n assignments again. I hope it could push me to revise those classical algorithms, and go beyond than just knowing interfaces of PyTorch or TensorFlow.

## Plans

I will finish main parts of this assignments, and writing down what I learnt in such a learning process. Here "main parts" means frequently used parts like convolutional layer implementation, batch normalization implementation, dropout, and so on. Other parts like PyTorch/TensorFlow API study as I've known about it already, or parts like GAN, Style Transfer as they lie in a very specific region in computer vision, will not be implemented.

So in general, I'll only finish all of [assignment 1](https://github.com/ECer23/cs231n.assignments/tree/master/assignment1) and most of [assignment 2](https://github.com/ECer23/cs231n.assignments/tree/master/assignment2) (except for frameworks parts). In the future, I might finish other parts.
