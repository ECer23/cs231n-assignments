## Data

Proper data setting before and during training could heavily boost training process, as we handily remove some undesired properties of data distribution while keep the features we want.

### Preprocessing

There are three common forms of data preprocessing a data matrix X

1. **Mean subtraction**, which centers the cloud of data around the origin along every dimension
2. **Normalization**, which sets data dimensions to the same scale
3. **PCA and Whitening**, which decorrelates data and make them has unit variance

In practice, PCA/Whitening and normalization are not used with Convolutional Networks. However, it is very important to zero-center the data. Because in principal, the linear transformation performed by PCA can be performed just as well by by the input layer weights of the neural network, so it isn't strictly speaking necessary. And in case of images, the relative scales of pixels are already approximately equal (and in range from 0 to 255), so it is not strictly necessary to perform this additional preprocessing step.

### Batch Normalization

The most important data processing used in training is batch normalization. It's highly recommended to read the original paper [here](https://arxiv.org/abs/1502.03167). The motivation of BN is called internal covariate shift. In human's words, data distribution will change during training, while we want to control the distribution more "normal" or "Gaussian", which could release some the weights initialization's efforts. So the straightforward way is to "make it normal".

In the implementation, applying this technique usually amounts to insert the BN layer immediately after FC layers. [Here](http://ecr23.me/vision/bn/) is a another blog I wrote that introduces BN. You can read it if you're not very familiar with the details of it.

In a nutshell, what we need to remember is that:

1. BN is a very useful tool and could be applied to your ConvNets in most of the times. Based on my own experiences, BN usually will increase accuracy around 2%-3%.
2. BN computes running mean and variance during training, which are used in inference time. And during inference time, mean and variance for test batch **will NOT** be computed.
3. BN is sensitive to small batch size because if the batch size is small, sample mean may contain a lot of noise. So try to enlarge the batch size, or try alternatives like layer normalization.

## Regularization

Regularization is used to control the capacity of Neural Networks to prevent over-fitting. 

### L1 and L2 regularization

**L2 regularization**, or commonly known as weight decay, is perhaps the most popular form of regularization. It can be implemented by penalizing the squared magnitude of all parameters directly. The L2 regularization has the intuitive interpretation of heavily penalizing peaky weight vectors and preferring diffuse (smaller and simpler) weight vectors. Also notice that during gradient descent parameter update, using the L2 regularization ultimately means that every weight is decayed linearly towards zero.

![](http://www.chioka.in/wp-content/uploads/2013/12/least_squares_l2.png)

**L1 regularization** has the intriguing and fascinating property that it leads the weight vectors to become very close to exactly zero. In other words, neurons with L2 regularization end up using only a sparse subset of their most important inputs as most weight goes very close to zero and become nearly invariant to the "noisy" inputs. In comparison, final weight vectors from L2 regularization are usually diffuse, small numbers. The sparsity property induced by L1 regularization has been used extensively as a feature selection mechanism.

![](http://www.chioka.in/wp-content/uploads/2013/12/least_squares_l11.png)

In practice, if you are not concerned with explicit feature selection, L2 regularization can be expected to give superior performance over L1. You may feel confused why L1 regularization could "select features". That's because L1 regularization updates weights by a scalar speed, while L2 regularization updates weights by a linear speed. It means that when weights are very small, L2 will slow down regularizing it while L1 will keep the speed and only leaves some very important weights. So L2 regularization will keep the weights small but not sparse, while L1 will make the weights sparse (a lot of zeros).

### Dropout

### Bias regularization ?

We typically penalize only the weights of the affine transformation at each layer and leaves the biases unregularized. There're 2 reasons to do so:
* Biases usually require less data to ﬁt accurately than the weights, which means they don't induce too much variance even if they're unregularized.
* Regularizing the bias parameters can introduce a signiﬁcant amount of under-fitting. 

## References

* [Neural Networks Part 2: Setting up the Data and the Loss](http://cs231n.github.io/neural-networks-2/) from Stanford CS231n course notes
* [Why do ML algorithms work better with uncorrelated data? What happens when we normalize the features?](http://qr.ae/TUGzs5) by Prasoon Goyal
* [Does Neural Networks based classification need a dimension reduction?](https://stats.stackexchange.com/questions/67986/does-neural-networks-based-classification-need-a-dimension-reduction)
* [Batch Normalization Notes](http://ecr23.me/vision/bn/) by Ecr23
* [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167) by Sergey Ioffe, et al.
* [Deep Learning: Regularization Notes](https://towardsdatascience.com/deep-learning-regularization-notes-29df9cb90779) by Tushar Gupta
* [Differences between L1 and L2 as Loss Function and Regularization](http://www.chioka.in/differences-between-l1-and-l2-as-loss-function-and-regularization/)
* [Dropout: A Simple Way to Prevent Neural Networks from Overfitting (pdf)](http://jmlr.org/papers/volume15/srivastava14a.old/srivastava14a.pdf) by N Srivastava, et al.
* [Dropout Notes](http://ecr23.me/vision/dropout/) by Ecr23