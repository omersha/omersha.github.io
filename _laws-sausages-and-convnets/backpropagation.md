---
layout: laws-sausages-and-convnets
title: "2. Backpropagation"
---
 
## 2.1. Convolutional Layers
<div><p>
I assume the reader is already familiar with backpropagation, but still, let's
start by taking a step backwards and think about backpropagation in the large.
The key principle underlying backpropagation is that a derivation is the arrow-
part of a covariant functor from pointed differentiable manifolds to linear
spaces. A.k.a, the chain rule. In the introduction I wrote that "backpropagation
is a modular reformulation of the chain rule". It's time to elaborate.
</p></div>

<div><p>
Consider the two functions $g:R^{N_2}\rightarrow R^{N_1}$ and
$f:R^{N_1}\rightarrow R$, and freely assume differentiability wherever it's
needed: according to the chain rule, the differential of the composition $f\circ
g:R^{N_2}\rightarrow R$ is given by $\nabla(f\circ g)=J_f\circ\nabla g$ (where
$J_f$ is the Jacobian of $f$) - or in explicit coordinates, for all $0\le i\lt
N_2$ we have $\frac{d(f\circ
g)}{dx_i}=\sum_{j=0}^{N_1-1}\frac{df}{dx_j}\frac{dg}{dx_i}$.
</p></div>

<div><p>
In machine learning, typical learning algorithms involve a functional model
$M:R^{N_I}\times R^{N_W}\rightarrow R^{N_O}$ and a loss-function
$E:R^{N_O}\rightarrow R$, and are eventually reduced to an optimization of
$E\circ M:R^{N_I}\rightarrow R$. The symbols $I,O,W$ and $E$ (for <b>I</b>nputs,
<b>O</b>utputs, <b>W</b>eights and <b>E</b>rror) are going to be used frequently from now
on, and we treat $O:=M(I;W)$ as a function of $I\in R^{N_I}$ and $W\in R^{N_W}$,
and $E(O)$ as a function of $O\in R^{N_O}$.
</p></div>

<div><p>
When the dimensions are high and the functions involved are sufficiently smooth,
the best optimization strategies make use of the gradient $\frac{d(E\circ
M)}{dW}$. Of course, non-smooth functions play a very important role in machine
learning - but at least in principle, all of this works just as well with
Lipschitz continuity and proximal-subgradients. By the chain rule, we can
succinctly write: $\frac{dE}{dW}=\frac{dE}{dO}\frac{dO}{dW}$.
</p></div>

<div><p>
Now, if $M$ itself is a composition $M:=M_\text{outer}\circ M_\text{inner}$,
then $\frac{dE}{dW_\text{inner}}=\frac{dE}{dO_\text{inner}}\frac{dO_\text{inner}
}{dW_\text{inner}}$. But $O_\text{inner}=I_\text{outer}$, so the only contextual
information $M_\text{inner}$ needs is $\frac{dE}{dI_\text{outer}}$. In its core,
backpropagation is simply the recursive application of this idea. From a
software-engineering perspective, it means it's possible to represent any
(differentiable) real-valued composite function as a tree whose nodes are the
functions' components, and for computing the evaluation of either the function
or its gradient all the nodes can work locally. So each node can have many
vector inputs $I_1,..., I_k$, and it should provide 2 algorithms (with a uniform
interface across the nodes):
</p></div>

<div><p>
<ol>
<li> A function that computes (locally) $(I_1,..., I_k)\mapsto O$, referred to as
the "Forward Algorithm".</li>
<li> A function that computes (locally) $(\frac{dE}{dO},
I_k)\mapsto\frac{dE}{dI_k}$ for each $k$, referred to the "Backward Algorithm".</li>
</ol>
</p></div>

<div><p>
There are several issues that deserve some more attention. First, nodes may be
either parameterized or not. If they are, and those parameters are to be a
subject of optimization, then the node should provide additional function that
computes ($\frac{dE}{dO}, I)\mapsto\frac{dE}{dW}$. This is usually treated as
part of the backward algorithm.
</p></div>

<div><p>
Secondly, the output of a node can be easily directed to several other nodes.
In this case, its $\frac{dE}{dO}$ would be an accumulation of all the $\frac{dE}{dI}$s
from its outgoing neighbours (since gradients are linear).
</p></div>

<div><p>
Thirdly, it is often very useful to express the backward
algorithm in terms of the output $O$ instead of the inputs $I$. In practice,
it's often rather simple to keep those values around after computing them in the
forward-algorithm, and it can save a lot of work when executing the backward
algorithm.
</p></div>

<div><p>
As an illustration, consider the very simple $R\mapsto R$ node whose forward
algorithm computes $\omega x^2$, where $\omega$ is considered a parameter. Then
the backward algorithm takes $\frac{dE}{dO}\in R$ as an input, and computes
$\frac{dE}{dI}=2\omega\frac{dE}{dO}x=2\omega\frac{dE}{dO}I$ and
$\frac{dE}{dW}=\frac{dE}{dO}x^2=\frac{dE}{dO}\omega^{-1}O$.
</p></div>

<div><p>
As for a more complicated example - well, the whole post is about one such
example. Convolutional layers are functions that take an input $I$ and a kernel
$W$, and in their forward algorithm computes the convolution $I\ast W$: 
</p></div>

**In [1]:**

{% highlight python %}
def convolve(A, B):
    return np.convolve(A, B, mode='valid')

def forward(W, I):    
    return convolve(I, W)
{% endhighlight %}
 
<div><p>
Next, the backward algorithms of convolutional layers will be discussed.
Efficient algorithms and implementations for both the forward and backward
algorithms for such functions are the subject of all of the following post.
</p></div>

<div><p>
Before going into details, a technical note: direct testing of an implementation
for the backward algorithm is possible by comparing its results to a numerical
differentiation of the forward algorithm. This procedure is known as <b>gradient
checking</b>. Numerical differentiation is a pretty huge subject by its own right,
but for the purpose of correctness tests, we can get by with a basic central
finite-difference approximation. Python has an out-of-the-box implementation of
it, given by <code>scipy.optimize.check_grad</code>. It works roughly as following: 
</p></div>

**In [2]:**

{% highlight python %}
def gradient_checking(func, x, index, epsilon=1e-6):
    x_curr = x.copy()
    x_curr[index] += epsilon
    err1 = func(x_curr)
    x_curr = x.copy()
    x_curr[index] -= epsilon
    err2 = func(x_curr)
    return (err2-err1)/(2*epsilon)
{% endhighlight %}
 
<div><p>
An indirect sanity check for an implementation of the backward algorithm, which
is certainly less reliable but more closely related to applications, is to use
it for optimization. Python of course has a not-too-bad (not too-good either)
optimization library as part of <code>scipy</code>, but I prefer using homemade toy SGD
implementations for debugging. They can provide more insight on what's going on
when things go south: 
</p></div>


**In [3]:**

{% highlight python %}
# Arguments:
#    W              : Weights (vector)
#    I              : Inputs (vector)
#    target         : Desired output (vector)
#    forward        : function(W, I) for the forward algorithm
#    supervise      : function(O, target) that returns the error gradient dEdO
#    backward_dEdW  : function(I, dEdO) for the backward algorithm (weights)
#    backward_dEdI  : function(W, dEdO) for the backward algorithm (inputs)
#    iters          : Number of iterations (iteger)
#    rate           : Learning rate (between 0 to 1)


def dummy_sgd_weights(W, I, target, forward, supervise, backward_dEdW, iters, rate):
    for i in xrange(iters):
        err, dEdO = supervise(forward(W, I), target)
        dEdW = backward_dEdW(I, dEdO)
        W -= rate*dEdW/np.max(np.abs(dEdW))
    return W

def dummy_sgd_inputs(W, I, target, forward, supervise, backward_dEdI, iters, rate):
    for i in xrange(iters):
        err, dEdO = supervise(forward(W, I), target)
        dEdI = backward_dEdI(W, dEdO)
        I -= rate*dEdI/np.max(np.abs(dEdI))
    return I
{% endhighlight %}
 
<div><p>
As for the loss function, we shall use MSE for debugging. The loss is given by
$E(O,T)\propto ||O-T||$, and so its gradient is given by $\frac{dE}{dO}\propto
2(O-T)$: 
</p></div>

**In [4]:**

{% highlight python %}
def supervise(O, target):
    error = np.mean(np.square(O-target))
    dEdO = 2.0*(O-target)/(len(O)+0.0)
    return error, dEdO
{% endhighlight %}
 
## 2.2. The Backward Algorithm
<div><p>
Let's start with the inputs. From the chain rule,
$\frac{dE}{dI}=\frac{dE}{dO}\frac{dO}{dI}$. The outputs are related to the
inputs via linear convolution: the $n$-th output is $O_n=(I\ast
W)_n:=\sum_{k=-M}^{+M}I_{n-k}W_{M+k}$. Thus $\frac{dO_n}{dI_k}=0$ whenever
$|n-k|>M$, and otherwise, $\frac{dO_n}{dI_k}=W_{M+n-k}$. This implies that $\frac{dE}{dI_k}=\sum_{i=0}^{2M}\frac{dE}{dO_{k-M+i}}\frac{dO_{k-M+i}}{dI_k}=\sum_{i=
0}^{2M}\frac{dE}{dO_{k-M+i}}W_i$, and kinda gives an algorithm: each element of
the gradient is computed via a dot-product: 
</p></div>

**In [5]:**

{% highlight python %}
def kinda_backward_inputs(W, dEdO):
    N = len(dEdO)
    M = (len(W)-1)/2
    dEdO = np.pad(dEdO, (M, M), mode='constant', constant_values=0.0)
    dEdI = np.zeros(N)
    for i in xrange(M,N+len(W)):
        dEdI[i] = np.dot(dEdO[i-M:i+M], W)
    return dEdI
{% endhighlight %}
 
<div><p>
As the term "kinda" gently hints, there a better perspective. Recall that $O_n$
can also be expressed as a dot-product: $O_n:=(I\ast W)_n=\langle I_{n-M}^{n+M},
W_\rho\rangle$ where $I_{n-M}^{n+M}$ is the projection of $I$ on $C^{2M+1}$
given by <code>I[n-m:n+m+1]</code> and $W_\rho$ is $W$ "reversed".
</p></div>

<div><p>
Looking back on what we have just done, we see that $\frac{dE}{dI}$ is actually
a result of a cross-correlation, and can be computed via a convolution:
$\frac{dE}{dI}=\frac{dE}{dO}\star W=\frac{dE}{dO}\ast W_\rho$. That's great! The
formulation of the backwards algorithm in terms of the forward algorithm will
allow us to reuse a single efficient implementation for both: 
</p></div>

**In [6]:**

{% highlight python %}
def backward_dEdI(W, dEdO):
    M = len(W)-1
    return forward(np.pad(dEdO, (M, M), mode='constant', constant_values=0.0), W[::-1])
{% endhighlight %}
 
<div><p>
Can we do the same for the weights? Again, we start with the chain rule,
$\frac{dE}{dW}=\frac{dE}{dO}\frac{dO}{dW}$, and consider $\frac{dO_n}{dW_k}$.
This time we obtain $\frac{dO_n}{dW_k}=I_{n+M-k}$, thus: $\frac{dE}{dW_k}=\sum_{
i=0}^{N-1}\frac{dE}{dO_i}\frac{dO_i}{dW_k}=\sum_{i=0}^{N-1}\frac{dE}{dO_i}I_{i+M
-k}$. Note that by applying a similar reasoning, then denoting by $\rho[I]$ a
padded and reversed version of $I$, this means that
$\frac{dE}{dW_k}=\sum_{i=0}^{N+2M}\frac{dE}{dO_i}\rho[I]_{N+k-i}$, so
$\frac{dE}{dW}=\frac{dE}{dO}\star I=\frac{dE}{dO}\ast \rho[I]$. Success!
</p></div>

<div><p>
Note that this time, this is a convolution ("valid", not "full", in numpy-lingo)
of two long sequences, and not a  convolution of one long sequence with another
short sequence as before: 
</p></div>

**In [7]:**

{% highlight python %}
def backward_dEdW(I, dEdO):
    return forward(dEdO, I[::-1])
{% endhighlight %}
 
<div><p>
To verify that we're on the right track, let's first test our functions with
gradient-checking: 
</p></div>

**In [8]:**

{% highlight python %}
I0 = np.random.normal(0.0, 1.0, 1000)
W0 = np.random.normal(0.0, 1.0, 50)
target = forward(np.random.normal(0.0, 1.0, 1000), W0)


check_dEdI = scipy.optimize.check_grad(func=lambda I: supervise(forward(I, W0), target)[0],
                                       grad=lambda I: backward_dEdI(W0, supervise(forward(I, W0), target)[1]),
                                       x0=I0)

check_dEdW = scipy.optimize.check_grad(func=lambda W: supervise(forward(I0, W), target)[0],
                                       grad=lambda W: backward_dEdW(I0, supervise(forward(I0, W), target)[1]),
                                       x0=W0)

print 'Gradient Checking:'
print '\t dEdI = %f' % check_dEdI
print '\t dEdW = %f' % check_dEdW
{% endhighlight %}

**Out [9]:**
<pre>
    Gradient Checking:
    	 dEdI = 0.000026
    	 dEdW = 0.000010
</pre>
 
<div><p>
And finally, let's try to use those gradients in a toy-optimization problem. If
our derivation is correct, we should able to make it work. First, minimizing the
error with respect to the weights: 
</p></div>

**In [10]:**

{% highlight python %}
R = 2
W = 2-np.power(np.linspace(-1, 1, R*2+1), 2)
I = np.cumsum(np.random.normal(0.0, 1.0, 100))

target = forward(np.ones(2*R+1)/(1.0+2*R), I)
W_found = dummy_sgd_weights(W.copy(), I, target, forward, supervise, backward_dEdW, iters=25, rate=0.1)

print 'Weights MSEs:'
print '\t Before: ', np.mean(np.square(forward(W, I)-target))
print '\t After:  ', np.mean(np.square(forward(W_found, I)-target))
{% endhighlight %}

**Out [10]:**
<pre>
    Weights MSEs:
    	 Before:  1231.53055406
    	 After:   1.60601207261
</pre>

 
<div><p>
Then, minimizing the error with respect to the inputs: 
</p></div>

**In [11]:**

{% highlight python %}
R = 10
W = np.random.normal(0.0, 1.0, 2*R+1)
I = np.random.normal(0.0, 1.0, 100)

target = forward(W, np.random.normal(0.0, 1.0, 100))
I_found = dummy_sgd_inputs(W, I.copy(), target, forward, supervise, backward_dEdI, iters=25, rate=0.1)

print 'Inputs MSEs:'
print '\t Before: ', np.mean(np.square(forward(W, I)-target))
print '\t After:  ', np.mean(np.square(forward(W, I_found)-target))
{% endhighlight %}

**Out [11]:**

    Inputs MSEs:
    	 Before:  29.402793521
    	 After:   2.89250420581


## 2.3. Backpropagation in the Frequency Domain
<div><p>
A natural thought that might have occurred to you, is the following: when
implementing a convolutional layer via a spectral algorithm, it seems redundant
to constantly transform and then inversely transform the weights for each train
sample (or a batch). It's reasonable to guess that by maintaining the weights in the
frequency domain, and computing the gradient of the error with respect to the
transformed weights, we can save some work at each iteration, and possibly
meaningfully accelerate the whole training algorithm.
</p></div>

<div><p>
I may be wrong about it, but unfortunately, this doesn't seem to work. If you
think otherwise, I'd be happy hear why and how. My reasoning is this: denoting
$\hat{W}:=\mathcal{F}(W)$, we have $O_n=(I\ast W)_n=(I\ast \mathcal{F}^{-1}(\hat
{W}))_n=\sum_{k=-M}^{+M}I_{n-k}\mathcal{F}^{-1}(\hat{W})_{M+k}$. Since by
definition $\mathcal{F}^{-1}(\hat{W})_k=\frac{1}{2M+1}\sum_{n=0}^{2M}\hat{W}_ne^
{\frac{i2\pi kn}{2M+1}}$, we obtain
$O_n=\frac{1}{2M+1}\sum_{k=-M}^{+M}I_{n-k}\sum_{r=0}^{2M}\hat{W}_re^{\frac{i2\pi
(M+k)r}{2M+1}}$.
</p></div>

<div><p>
Denoting by $\rho X$ the reversal of $X$, this implies:

$$\frac{dO_n}{d\hat{W}_r}=\frac{1}{2M+1}\sum_{k=-M}^{+M}I_{n-k}e^{\frac{i2\pi (M+k)r}{2M+1}}=\frac{1}{2M+1}\sum_{k=0}^{2M}I_{n-k+M}e^{\frac{i2\pi kr}{2M+1}}=\mathcal{F}^{-1}(\rho I_{n-M}^{n+M})_r$$

and we conclude: $\frac{dE}{d
\hat{W}_r}=\sum_i\frac{dE}{dO_i}\frac{dO_i}{d\hat{W}_r}=\sum_i\frac{dE}{dO_i}\mathcal{F}^{-1}(\rho I_{i-M}^{i+M})_r$.
</p></div>

<div><p>
The result is kinda weird (note the application of an inverse Fourier transform
for a spatial sequence), so let me assure myself that no silly mistakes - as I
so often produce - were involved here, by a gradient-checking of a grotesquely
unoptimized implementation of this algorithm: 
</p></div>

**In [12]:**

{% highlight python %}
def spectral_forward(I, W_transformed):
    return forward(np.real(np.fft.ifft(W_transformed)), I)

def spectral_backward_dEdW(I, dEdO):    
    R = len(I)-len(dEdO)
    res = np.zeros(R+1, np.complex128)
    for index in xrange(R+1):
        for i in xrange(len(dEdO)):
            res[index] += np.real(np.fft.ifft(I[i:i+R+1][::-1]))[index]*dEdO[i]
    return np.real(res)


I0 = np.random.normal(0.0, 1.0, 1000)
W0 = np.fft.fft(np.random.normal(0.0, 1.0, 50))
target = spectral_forward(np.random.normal(0.0, 1.0, 1000), W0)

check_dEdW = scipy.optimize.check_grad(func=lambda W: supervise(spectral_forward(I0, W), target)[0],
                                       grad=lambda W: spectral_backward_dEdW(I0, supervise(spectral_forward(I0, W), target)[1]),
                                       x0=W0)
print check_dEdW
{% endhighlight %}

**Out [13]:**
<pre>
    3.69439780792e-06
</pre>
 
<div><p>
So it seems correct, which is actually not happy news. Instead of 1
transformation-inversion pair of size $2M+1$ that takes $O(M\log{M})$, we now
need N transformation-inversion pairs of that size, which takes $O(NM\log{M})$.
Yuck. The implementation above actually does many more than that, but it is
grotesquely unoptimized  by design.
</p></div>

<div><p>
The root of all evil seems to be the fact the insistence of working directly
with $\frac{dE}{dO}$, which is the fundamental input of a differential node.
Apparently, expressing $\frac{dO_n}{d\mathcal{F}(W)_r}$ as a function of
$\frac{dE}{dO}$ leads to a terribly inefficient algorithm. Unless, of course,
there is some clever way I missed for quickly calculating the blob of weirdness
$\sum_i\frac{dE}{dO_i}\mathcal{F}^{-1}(\sigma I_{i-M}^{i+M})_r$.
</p></div>

<div><p>
Alternatively, we can work "fully" in the frequency domain: since
$\mathcal{F}(O)_n=\mathcal{F}(I)_n\cdot\mathcal{F}(W)_n$, we get that
$\frac{d\mathcal{F}(O)_n}{d\mathcal{F}(W)_r}=0$ if $r\neq n$, and otherwise
$\frac{d\mathcal{F}(O)_n}{d\mathcal{F}(W)_r}=\mathcal{F}(I)_n$. This implies
that $\frac{dE}{\mathcal{F}(W)_r}=\sum_i\frac{dE}{d\mathcal{F}(O)_i}\frac{d\mathcal{F}(O)_i}{d\mathcal{F}(W)_r}=\frac{dE}{d\mathcal{F}(O)_r}\mathcal{F}(I)_r$.
</p></div>

<div><p>
So $\frac{dE}{d\mathcal{F}(O)_r}=\sum_n{(\frac{dE}{dO_n}\frac{dO_n}{d\mathcal{F}
(O)_r})}=\frac{1}{N-2M}\sum_n{(\frac{dE}{dO_n}e^{\frac{i2\pi
nr}{N-2M}})}=\mathcal{F}^{-1}(\frac{dE}{dO})_r$, and thus
$\frac{dE}{\mathcal{F}(W)_r}=\mathcal{F}^{-1}(\frac{dE}{dO})_r\mathcal{F}(I)_r$.
That's much better than the previous attempt, but not better than "regular"
backpropagation: it is still $N\log{N}$, since we must calculate
$\mathcal{F}^{-1}(\frac{dE}{dO})$.
</p></div>

<div><p>
There's nothing counter intuitive about it: $\frac{dE}{dW}$ is a function of
$\frac{dE}{dO}$ and $I$. Even if the convolutional layer readily has
$\mathcal{F}(I)$ (as an intermediate step of the forward algorithm), it still
doesn't have $\mathcal{F}(\frac{dE}{dO})$ (if needed, it must be calculated from
$\frac{dE}{dO}$). So either way, some DFT must be performed, which means the
backwards algorithm for the weights will be always be superlinear with respect
to $N$.
</p></div>

<div><p>
And I'm forced to conclude that the seemingly cool idea of backpropagation in
the frequency domain is not helpful. 
</p></div>


## 2.4. Concurrent Training
The context in which backpropagation is useful, is learning via gradient-based
optimization. Such optimization algorithms are pretty much universally iterative.
Thus in principle, learning with backpropagation is a sequential task, and very
latency-sensitive. The faster a single sample can be processed, the faster the
overall training would be.

But concurrency can still be utilized in several important ways. First and foremost,
concurrent algorithms for computing the gradient associated with a single sample
can significantly speed-up the training process. In the following, much attention
will be given to concurrent algorithms for fast convolutions.

Secondly, using batch-training can utilize concurrency. Typical mini-batches consist
of 10 to 500 samples, which may be processed in parallel (using fixed values for the
model's parameters). Batches are especially useful when employing quasilinear
algorithms (to be discussed soon). This has to do with the restriction to real values,
and the fact convolutions can be done by using out-of-order Fourier-transforms.

Thirdly, there are clever "tricks" that can be used to utilize concurrency for training.
Those include model averaging, genetic algorithms and variations of ensemble methods.
Those won't be treated here.
