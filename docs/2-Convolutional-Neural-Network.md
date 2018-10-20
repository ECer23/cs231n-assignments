# Convolutional Nerual Network

## Architecture Design

### Why choose ConvNet over regular NN

Regular NN don’t scale well to full images, or in other words, they don't fit very well for images' structure as images have width, height and channels. For image with size 200x200x3, the neurons will have 120,000 weights. Full connectivity is wasteful and the huge number of parameters would quickly lead to over-fitting.

ConvNets take advantage of the images structure and try to reserve information inside the structure. The layers of a ConvNet have neurons arranged in 3 dimensions: width, height, depth. In short, A ConvNet is made up of layers and every Layer has a simple API: **It transforms an input 3-dimension volume to an output 3-dimension volume**. For back propagation's sake, each layer must be some differentiable function that may or may not have parameters. 

### Convolutional layer

The Convolutional layer is the core building block of a Convolutional Network that does most of the computational heavy lifting. 

**Local connectivity**. Every filter is small spatially (along width and height), but extends through the full depth of the input volume.  As we slide the filter over the width and height of the input volume we will produce a 2-dimensional activation map that gives the responses of that filter at every spatial position. As we saw above it is impractical to connect neurons to all neurons in the previous volume. Instead, we will connect each neuron to only a local region of the input volume. The spatial extent of this connectivity is a hyper-parameter called the receptive field of the neuron (equivalently this is the filter size). Each neuron in the convolutional layer is connected only to a local region in the input volume spatially, but to the full depth (i.e. all color channels).

![](http://cs231n.github.io/assets/cnn/depthcol.jpeg)

**Parameter sharing**. Parameter sharing scheme is used in Convolutional Layers to control the number of parameters. Parameter sharing scheme is used in Convolutional Layers to control the number of parameters. We assume that if one feature is useful to compute at some spatial position (x,y), then it should also be useful to compute at a different position (x2,y2). In other words, denoting a single 2-dimensional slice of depth as a depth slice, and weights for a depth slice are all the same.

### Activation functions comparisons

The general advice is to use ReLU or Leaky ReLU.

* **Sigmoid**
    * (-) Sigmoids saturate and kill gradients. A very undesirable property of the sigmoid neuron is that when the neuron’s activation saturates at either tail of 0 or 1, the gradient at these regions is almost zero
    * (-) Sigmoid outputs are not zero-centered. If inputs are not normalized, the gradients will always be positive or negative (like explained in image below)
    ![](http://wx2.sinaimg.cn/large/0060lm7Tly1fwcdwu5ol2j30nm07ywfc.jpg)
* **Tanh**
* **ReLU**
    * (+) It's linear, non-saturating so it's simple and fast
    * (-) ReLU units can be fragile if the inputs are less than zero.
* **Leaky ReLU**
    * (+) Fix ReLU's problem when inputs are < 0. It also gives a very small activation when inputs are less than zero.

## References

* [Convolutional Networks](http://cs231n.github.io/convolutional-networks/) from Stanford CS231n course notes
* [Commonly used activation functions](http://cs231n.github.io/neural-networks-1/#actfun) from Stanford CS231n course notes
* [Why are non zero-centered activation functions a problem in backpropagation?](https://stats.stackexchange.com/questions/237169/why-are-non-zero-centered-activation-functions-a-problem-in-backpropagation) from CrossValidated by dontloo