---
layout: page
title: "Laws, Sausages and ConvNets"
preview: "The nuts and bolts of Convolutional Neural Networks: algorithms, implementations and optimizations."
---
> Laws, like sausages, cease to inspire respect in proportion as we know how
they are made.

[Convolutional Neural
Networks](https://en.wikipedia.org/wiki/Convolutional_neural_network) (CNNs or
ConvNets in short) give the state-of-the-art results in many problem domains.
Nowadays, using them is simpler than ever: pretty much all the frameworks out
there (<a href="http://deeplearning.net/software/theano/">Theano<a>,
<a href="http://torch.ch/">Torch</a>, <a href="http://caffe.berkeleyvision.org/">Caffe</a>...)
support them out-of-the-box. If you care only about applying such networks -
move along then, there's nothing to see here. This post is about the nuts and
bolts: algorithms, implementations and optimizations. It is about how ConvNets are made.

Full-blown ConvNets may incorporate a variety of ideas and mechanisms, but in
the following I'm going to focus on their very core: convolutional layers.
[Convolution](https://en.wikipedia.org/wiki/Convolution) is a simple
mathematical operation, so the enormous complexity involved in implementing
convolutional layers may be surprising. The fun begins due to the interaction of
mathematical and algorithmic considerations with real-world constraints
regarding parallelism, memory-hierarchies and heterogeneous systems. And then -
to really get the party started - there's the question of numerical accuracy.

This post is long and rich in details. Perhaps some snippets, worth
highlighting, will get dedicated short posts of their own in the future. Python
will be used for algorithmic prototyping. ```C``` and some ```C++```, together
with [OpenCL](https://www.khronos.org/opencl/) (which I prefer over ```CUDA```),
will be used for implementations. Concurrency is going to get a lot of
attention, mostly in the context of heterogeneous computing (e.g. ```GPU```s and
alike).

The general spirit of things to come is perhaps somewhat unorthodox within the
deep-learning community. It is very common to introduce and think about
convolutional layers in terms of neural connectivity and weights-sharing, e.g.
as in this diagram (taken from [here](https://cp4space.wordpress.com/2016/02/06
/deep-learning-with-the-analytical-engine/), and chosen for its aesthetic
appeal):

<figure>
<img src="{{ site.baseurl }}/assets/laws-sausages-and-convnets_files/convnet-example.png" style='width: 750px; margin-left: auto; margin-right: auto;'/>
</figure>

I personally think that weights-sharing is a horrible way to describe and
understand CNNs, and it really misses the point of what backpropagation is all
about: [automatic
differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation) of
real-valued functions. Instead of dealing with networks, I take the point
of view that a convolutional layer is simply a differentiable function. There's
more to it than that, of course: backpropagation is a modular reformulation of
the chain rule, and the  differentiable function that model a convolutional
layer needs to be reformulated as a differentiable node. This will be explained in
details.

While writing this overview, I discovered that writing is... well, hard. Proofreading,
editing and phrasing are never-ending tasks, and as such, I'm not done with them. Some
parts of this text are still rather crude, and I will likely continue to edit it long
after its publication. Corrections, suggestions and notes are most welcomed!
 
### Content

1. [Convolutions]({{ site.baseurl }}/laws-sausages-and-convnets/convolutions.html)
    1. Feature Detectors
    2. Linear Convolution and Cross-Correlation
    3. The Discrete Fourier Transform
    4. Circular Convolution and Spectral Interpolation

2. [Backpropagation]({{ site.baseurl }}/laws-sausages-and-convnets/backpropagation.html)
    1. Convolutional Layers
    2. The Backward Algorithm
    3. Backpropagation in the Frequency Domain
    4. Concurrent Training

3. [Quasilinearity]({{ site.baseurl }}/laws-sausages-and-convnets/quasilinearity.html)
    1. Cooley-Tukey and Good–Thomas
    2. Bluestein, Rader and Singleton
    3. Decimation in Time and Frequency
    4. Transpositions and Permutations
    5. Execution Plans
    6. Quasilinear Convolutions

4. [Overlap Methods]({{ site.baseurl }}/laws-sausages-and-convnets/overlap-methods.html)
    1. Parallelized Filters
    2. Overlap-Discard
    3. Overlap-Add
    4. Convolution in Parts

5. [Multidimensionality]({{ site.baseurl }}/laws-sausages-and-convnets/multidimensionality.html)
    1. Separability
    2. The Row-Column Method and The Helix Transform
    3. Image Memory Objects
    4. Shared-Memory Bank Conflicts

