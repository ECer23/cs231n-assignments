# Classifier

## Table Of Contents

* [kNN Classifier](https://github.com/ECer23/cs231n.assignments/wiki/Classifier#knn-classifier)
    * [Decision boundary](https://github.com/ECer23/cs231n.assignments/wiki/Classifier#decision-boundary)
    * [L1 v.s. L2](https://github.com/ECer23/cs231n.assignments/wiki/Classifier#l1-vs-l2)
    * [Cross validation](https://github.com/ECer23/cs231n.assignments/wiki/Classifier#cross-validation)
    * [Advice on applying kNN](https://github.com/ECer23/cs231n.assignments/wiki/Classifier#advice-on-applying-knn)
* [Multi-class Support Vector Machine and Softmax Classifier](https://github.com/ECer23/cs231n.assignments/wiki/Classifier#multi-class-support-vector-machine-and-softmax-classifier)
    * [Linear Classifier](https://github.com/ECer23/cs231n.assignments/wiki/Classifier#linear-classifier)
    * [Differentiability](https://github.com/ECer23/cs231n.assignments/wiki/Classifier#differentiability)
    * [Cross entropy](https://github.com/ECer23/cs231n.assignments/wiki/Classifier#cross-entropy)
    * [Softmax function](https://github.com/ECer23/cs231n.assignments/wiki/Classifier#softmax-function)
    * [Regularization](https://github.com/ECer23/cs231n.assignments/wiki/Classifier#regularization)
* [Neural Networks](https://github.com/ECer23/cs231n.assignments/wiki/Classifier#neural-networks)
    * [Advice on training neural networks](https://github.com/ECer23/cs231n.assignments/wiki/Classifier#advice-on-training-neural-networks)
* [Implementation Details](https://github.com/ECer23/cs231n.assignments/wiki/Classifier#implementation-details)
    * [Vectorization](https://github.com/ECer23/cs231n.assignments/wiki/Classifier#vectorization)
    * [Matrix-Matrix multiply gradient](https://github.com/ECer23/cs231n.assignments/wiki/Classifier#matrix-matrix-multiply-gradient)
* [References](https://github.com/ECer23/cs231n.assignments/wiki/Classifier#references)

## kNN Classifier

### Decision boundary
**kNN is not a linear classifier**. Decision boundaries of kNN are composed of different pieces of lines like shown below. Stated in a more formal way, different data will fall in different regions, and these regions can't be separated by a hyperplane.

![](http://cs231n.github.io/assets/knn.jpeg)

### L1 v.s. L2
In particular, the L2 distance is much more unforgiving than the L1 distance when it comes to differences between two vectors. That is, the L2 distance prefers many medium disagreements to one big one. L1 and L2 distances (or equivalently the L1/L2 norms of the differences between a pair of images) are the most commonly used special cases of a p-norm

![](http://wx2.sinaimg.cn/large/0060lm7Tly1fw6p5twj4vj30xk0h3tek.jpg)

### Cross validation
For example, in 5-fold cross-validation, we would split the training data into 5 equal folds, use 4 of them for training, and 1 for validation. We would then iterate over which fold is the validation fold, evaluate the performance, and finally average the performance across the different folds

### Advice on applying kNN
1. Use kNN as baseline instead of real application
2. Preprocess your data: Normalize the features in your data (e.g. one pixel in images) to have zero mean and unit variance
3. If your data is very high-dimensional, consider using a dimensional reduction technique such as PCA
4. Split your training data randomly into train/val splits. As a rule of thumb, between 70-90% of your data usually goes to the train split.
5. If your kNN classifier is running too long, consider using an Approximate Nearest Neighbor library (e.g. FLANN) to accelerate the retrieval (at cost of some accuracy).

## Multi-class Support Vector Machine and Softmax Classifier

### Linear classifier

A linear classifier achieves this by making a classification decision based on the value of a linear combination of the characteristics. To train the linear classifier is to train the weights in linear combination. Unlike non-linear classifier like kNN, once we finish training, we can apply the weights on test data and leave the training data behind, which means the amount of training data won't influence inference time.

The geometric interpretation of these weights is that as each row will rotate corresponding line in the pixel space in different directions.

![](http://cs231n.github.io/assets/pixelspace.jpeg)

Another interpretation is that each row of weights matrix is a template for that class, and linear combination is doing a "template matching" work.

![](http://cs231n.github.io/assets/templates.jpg)

### Differentiability

the SVM loss function is not strictly speaking differentiable at hinge point, where `s_j - s_{y_i} = delta`. In the image below, you can see that the kinks in the loss function (due to the max operation) technically make the loss function non-differentiable because at these kinks the gradient is **not defined**.

![](http://cs231n.github.io/assets/svmbowl.png)

### Cross entropy
* Information: `I(x) = log(1/p(x))`. The more possible is something, the less information it contains
* Entropy: `E(x) = sum(p(x) * I(x))`. The randomness of a system. The entropy is large if many impossible things happen, which indicates this system is random.
* Cross Entropy: `H(p,q) = sum(p * I(q))`. It measures the expected entropy of predicted distribution q in true distribution p. Cross entropy is always bigger than original entropy of p, that's because predicted distribution introduces more randomness. And cross entropy loss is trying to penalize the randomness and make predicted distribution as close as true distribution.

### Softmax function
To compare with true distribution, all values in predicted distribution must lie in [0,1]. So we use softmax function to achieve this goal.

![](http://wx4.sinaimg.cn/mw690/0060lm7Tly1fwa32j4srzg304x01c0q4.gif)

### Regularization

There is one bug with the loss function we presented above. Suppose that we have a dataset and a set of parameters W that correctly classify every example. The issue is that this set of W is not necessarily unique: there might be many similar W that correctly classify the examples. One easy way to see this is that if some parameters W correctly classify all examples (so loss is zero for each example), then any multiple of these parameters where will also give zero loss because this transformation uniformly stretches all score magnitudes and hence also their absolute differences.

On the other hand, generalized linear classifiers usually prefer smaller weights. That's because smaller weights means features' variation play a more important role in linear combination. If the weights are too big, the classification work is heavily based on weights instead of features, and when the features changed a little, the outcome might shift to a long distance. Or in other words, the variance of outcome becomes very large with big weights. Data loss function can only make sure the bias of outcome is small, and regularization can help make the variance small.

## Neural Networks

### Advice on training neural networks

![](http://wx4.sinaimg.cn/large/0060lm7Tly1fwac27srjdj30hd0dsgmh.jpg)

Looking at the visualizations above, we see that the loss is decreasing more or less linearly, which seems to suggest that the learning rate may be too low. Moreover, there is no gap between the training and validation accuracy, suggesting that the model we used has low capacity, and that we should increase its size. On the other hand, with a very large model we would expect to see more over-fitting, which would manifest itself as a very large gap between the training and validation accuracy.

*Inline Question*: Now that you have trained a Neural Network classifier, you may find that your testing accuracy is much lower than the training accuracy. In what ways can we decrease this gap? Select all that apply.
1. Train on a larger dataset.
2. Add more hidden units.
3. Increase the regularization strength.
4. None of the above.

*Your explanation:*
1. Larger training sets can help reduce overfitting problems, provide more knowledge. While larger training sets mean it takes more time to train the neural network.
2. More hidden units will increase the complexity of neural networks. But sometimes more complex doesn't always mean better because the network might be too complex, which leads to overfitting problems
3. Suitable regularization strength will help reduce overfitting but if regularization is too strong, it might suppress the training of network and makes it difficult to converge, which also called under-fitting.

## Implementation Details

### Vectorization
It's the most difficult part for me in assignment 1. Take [linear_svm.py](https://github.com/ECer23/cs231n.assignments/blob/master/assignment1/cs231n/classifiers/linear_svm.py)'s gradients computation as an example. In `svm_loss_naive`, we compute the gradients in iterative way. 

```python
margin = scores[j] - scores[y[i]] + 1
if margin > 0:
  dW[:, j] += X[i]
  dW[:, y[i]] -= X[i]
```

According to the formula, it's easy to compute the gradients of `dW` in an iterative way. Here we notice that `X[i]` is a row vector with length D, and `dW[:,j]` is a column vector with length D, and the computation of column j in `dW` is related to the row number of `X`, which means that column j of `dW` is a linear combination of rows in `X`. So it's natural to think of matrix multiplication.

![](http://wx4.sinaimg.cn/mw690/0060lm7Tly1fwa27s56rfg307l016a9t.gif)

So the implementation for the vectorized version of svm loss function needs:

1. Compute the linear combination's weights
2. Compute the dot product between the weights and `X`
3. Perform matrix multiplication between `X` and `dW`

In the iterative version we can see that whether `X[i]` is added to `dW[:,j]` is related to the margin's value. If the margin is positive (which means it's a false classification for sample i), we should penalize correspond class's weights. And for true class `y[i]`, we need to encourage it's weights. Because gradient descents algorithms will update the weights to adverse direction, so "penalize" means "positive gradient" and "encourage" means "negative gradient".

```python
false_class = (margin > 0).astype(int)
false_class[np.arange(num_train), y] = -1 * np.sum(false_class, axis=1)
dW += X.T.dot(false_class.astype(int))
dW /= num_train
dW += 2 * reg * W
```

### Matrix-matrix multiply gradient
Example above is one special case of matrix-matrix multiply gradient. In [nerual_network.py](https://github.com/ECer23/cs231n.assignments/blob/master/assignment1/cs231n/classifiers/nerual_network.py), because in forward pass, it computes two matrix multiplications and we also need to compute matrix-matrix multiply gradient. Of course we can use math analysis methods like in SVM, but it's easier to analyze the dimensions. For scalar multiplication y = ax + b, dx = dy/a. When scalar becomes vector, we also want properties like that, but we found the matrix size is not consistent. So it's natural to think about "making a matrix".

```python
scores = hidden_out.dot(W2) + b2
dhidden = dscores.dot(W2.T) / N
dW2 = hidden_out.T.dot(dscores) / N
db2 = np.sum(dscores, axis=0) / N
```

In the code above, scores' shape equals `[N, C]`, hidden outputs' shape equals `[N, H]` and W2's shape equals `[H, C]`. dW2's shape is the same as W2, so to make a matrix like that size, we only need to perform `dscores.dot(W2.T)`. But here is not the end of the story, in matrix multiplication, we compute gradients over all examples given so we also have to average over example size to get the correct gradients. And that's the easiest way to compute Matrix-Matrix multiply gradient!

You might ask what does dscores mean. According to derivatives' chain rule, the end of the whole chain is y so dy = 1. And we use a softmax function to map scores to predicted labels y, so dscores comes from `dy/dscores`, and that's what we "back propagated" from softmax function.

## References

* [Nearest Neighbor Classifier](http://cs231n.github.io/classification/) from Stanford CS231n course notes
* [Linear Classifier](http://cs231n.github.io/linear-classify/) from Stanford CS231n course notes
* [Discussions about L1 distance](https://www.reddit.com/r/cs231n/comments/8yntcj/c231n_2018_assignment_1_inline_question_2_help/) on Reddit
* [Rotational Invariance](https://en.wikipedia.org/wiki/Rotational_invariance) on Wikipedia
* [如何通俗的解释交叉熵与相对熵？](https://www.zhihu.com/question/41252833/answer/108777563) from Zhihu.com, by Noriko Oshima