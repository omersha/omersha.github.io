---
layout: laws-sausages-and-convnets
title: "1. Convolutions"
---

### 1.1. Feature Detectors
<div><p>
Functionally, a convolution layer usually meant to act as a feature detector. In
other words, it is designed to identify signals within a noisy feed. There are
roughly three major variations on the theme of improving signal-to-noise ratio:
smoothing, change-detection and template-matching. Convolutions play a central
role in all three.
</p></div>

<div><p>
<b>Smoothing</b> is based on the assumption that the true signal effects similarly
adjacent samples while the effects of the noise on different samples are pretty
much statistically independent. Or alternatively, that the information carried
by the signal is concentrated in lower frequencies than the noise. Either way,
both perspectives lead to the idea of a filtration by some type of a moving-
average. A common approach is to weight each neighbour by its  similarity to the
point, as measured by a Gaussian radial function: 
</p></div>


**In [1]:**

{% highlight python %}
noisy_image = demo_image+np.random.normal(0.0,50.0, demo_image.shape)
x, y = np.mgrid[-4:5, -4:5]

plt.subplot(1,3,1)
plt.title('Original')
plt.imshow(demo_image, cmap = plt.get_cmap('gray'))
plt.subplot(1,3,2)
plt.title('Noisy')
plt.imshow(noisy_image, cmap = plt.get_cmap('gray'))
plt.subplot(1,3,3)
plt.title('Smoothed')
plt.imshow(scipy.signal.convolve2d(noisy_image, np.exp(-(x**2/3.0+y**2/3.0))), cmap = plt.get_cmap('gray'))
{% endhighlight %}

**Out [1]:**




![png]({{ site.baseurl }}/assets/laws-sausages-and-convnets_files/convolutions_5_1.png) 

 
<div><p>
In <b>change-detection</b>, the target signal is the change in the feed. There are
numerous applications for change-detections, such as numerical differentiation
and edge-detection. Compared to smoothing, this is a very different problem -
but with very similar solutions. This is because the derivative
$\lim_{h\rightarrow 0}\frac{f(x+h)-f(x)}{h}$ is a limit of a weighted average,
and the discrete approximations of it have the general form of weighted moving
averages. This time, the general computation of the weights is a bit
<a hred="https://amath.colorado.edu/faculty/fornberg/Docs/sirev_cl.pdf">tricky</a>, but
it's rarely an issue since the applicability of high-order finite differences is
limited anyway for the reason that round-off errors quickly shadow truncation
errors.
</p></div>

<div><p>
It's very common to apply smoothing as a preliminary step to change-detection,
since change detectors are sensitive to noise. In numerical differentiation
<a href="https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter">Savitzky-Golay</a> is
such a method, and in edge-detection, the <a href="https://en.wikipedia.org/wiki/Sobel_operator">Sobel operator</a>
is an example for this type of combination: 
</p></div>

**In [2]:**

{% highlight python %}
im_Gx = scipy.signal.convolve2d(demo_image, np.outer(np.array([1, 2, 1]), np.array([-1, 0, +1])))
im_Gy = scipy.signal.convolve2d(demo_image, np.outer(np.array([-1, 0, +1]), np.array([1, 2, 1])))

plt.subplot(1,3,1)
plt.title('Original Image')
plt.imshow(demo_image, cmap = plt.get_cmap('gray'))
plt.subplot(1,3,2)
plt.title('Change Magnitude')
plt.imshow(np.sqrt(np.square(im_Gx)+np.square(im_Gy)))
plt.subplot(1,3,3)
plt.title('Change Direction')
plt.imshow(np.arctan2(im_Gx, im_Gy))
{% endhighlight %}




![png]({{ site.baseurl }}/assets/laws-sausages-and-convnets_files/convolutions_7_1.png) 
 
<div><p>
Finally, in <b>template-matching</b> the feed is searched for approximated
occurrences of some predetermined template. For example, they can be used to
look for appearances of circle-like shapes in an image and alike. Such detectors
are often referred to as <a href="https://en.wikipedia.org/wiki/Matched_filter">matched filters</a>.
</p></div>

<div><p>
In general terms, the way they work is by performing a sliding inner-product of
the template and the feed. Up to normalization, inner products can be
interpreted as a similarity measure between two vectors. So a matched-filter
produces localized estimations for local-similarities to its template.
</p></div>

<div><p>
In general terms, the way they work is by performing a sliding inner-product.
As an example, let's pick an arbitrary symbol from the <a href="https://en.wikipedia.org/wiki/Voynich_manuscript">Voynich
manuscript</a>, and search for the 25 most similar symbols in those 2 pages: 
</p></div>

**In [3]:**

{% highlight python %}
plt.figure(figsize=(14, 14))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 4]) 

voynich_symbol = voynich_grey[384:397, 361:372].copy()
plt.subplot(gs[0])
plt.title('A symbol')
plt.imshow(voynich_symbol, cmap = plt.get_cmap('gray'))

xcorr = scipy.signal.correlate2d(voynich_grey-voynich_grey.mean(),
                                 voynich_symbol-voynich_symbol.mean(),
                                 boundary='symm', mode='same')
ys, xs = np.unravel_index(np.argsort(xcorr.flatten())[-25:], xcorr.shape)
plt.subplot(gs[1])
plt.imshow(voynich_color)
plt.plot(xs, ys, 'bo')
plt.title('Blue dot marks the spot')
{% endhighlight %}


 
![png]({{ site.baseurl }}/assets/laws-sausages-and-convnets_files/convolutions_9_1.png) 

<div><p>
All of the examples so far were linear, and they worked by convolving their feed
with some finite impulse response (also known as "kernel", but you can throw a
stone and hit 31 completely different things also known as "kernels").
Convolutions will be explained in detail in the next section.
</p></div>

<div><p>
Nonlinear feature-detectors are also a thing, but ConvNets can approximate
nonlinear feature-detectors in about the same way feed-forward networks can
approximate arbitrary nonlinear functions by chaining adjustable linear
functions with fixed nonlinear transformations (ConvNets add subsampling to this
mix). Anyway, at least for the meanwhile, nonlinear detectors seem to be much
less common in the context of ConvNets.
</p></div>

<div><p>
Functionally, a convolution layer works much like one of those filters above,
with one important twist: its impulse response is not "designed" or "engineered"
by some all-knowing "domain experts" (pfff). Instead, it is subject to
supervised learning. The novelty of employing them in the context of neural
networks, is in the fact that the usefulness of the learned impulse response is
measured by later layers that directly try to solve some given task.
Architecturally, convolutional layers are commonly hyper-parameterized by
<b>volume</b>, <b>stride</b> and <b>depth</b>.
</p></div>

<div><p>
Volume is a rather application-specific attribute that controls how a
convolutional layer treat a multi-channeled input. For example, color images may
come with 3-channels (Red, Green and Blue), and stereo audio may come in 2
channels (left and right). It is expected from a feature detector to "cut-
across" the channels and to produce a single univariate response for any
relevant spatial-temporal location. The usual approaches are either to filter
independently each channel and output cross-channel mean, or to employ a
multivariate filter and submit the weights of the different channels to the
learning process.
</p></div>

<div><p>
Stride is used to define a grid over the input on which the feature detection is
performed. This is equivalent (functionally, not computationally) to transverse
sub-sampling from the output of a stride-1 detector. This may help to reduce the
correlation between nearby outputs, but the real motivation for larger strides
is typically computational.
</p></div>

<div><p>
Depth refers to the applications of multiple different detectors on the same
input. The result of applying $K$ detectors is a $K$-dimensional vector per
location. The problem of dealing with those multivariate responses (called
"depth columns") is delegated to the following layer (which may employ sub-
sampling, smooth-transformations, other feature-detectors - or pretty much
anything that envisioned by the network designer). In principles, a non-
degenerate depth is the same as several unrelated convolutional layers, but in
practice those layers may (and should) share some common computations to fasten
things as much as possible.
</p></div>

<div><p>
Computationally, a feature-detector is parameterized by relatively few weights -
but is then applicable over arbitrarily large inputs. This is not only makes
them computationally attractive, but inherently contributes to the
regularization of the learning. 
</p></div>
 
### 1.2. Linear Convolution and Cross-Correlation
<div><p>
For simplicity, let's restrict the discussion to one-dimensional sequences for
the meanwhile. In the comfortable la-la land of infinite sequences, the
convolution $a\ast b$ of two sequences $a,b\in C^\infty$ is defined as $(a\ast b
)_n:=\sum_{k=-\infty}^{+\infty}a_kb_{n-k}=\sum_{k=-\infty}^{+\infty}a_{n-k}b_k$.
If only one of the sequences is finite, say $b\in C^{2M+1}$, things are still
pretty nice: $(a\ast b)_n:=\sum_{k=-M}^{+M}a_{n-k}b_{M+k}$.
</p></div>

<div><p>
When dealing with two finite sequences $a\in C^N$ and $b\in C^{2M+1}$ (assuming
w.l.g. that $N\gt 2M$)- which are the kind of sequences usually encountered in
practice - then $a\ast b\in C^{N-2M}$, and $(a\ast b)_n$ is an inner-product of
two vectors of length $M$: 
</p></div>

**In [4]:**

{% highlight python %}
# Direct, serial and inefficient linear convolution (using dot products):
def convolve(a, b): 
    longer = [a, b][np.argmax((len(a), len(b)))]
    shorter = [a, b][np.argmin((len(a), len(b)))]
    K = len(longer)-len(shorter)+1
    convolution = np.zeros(K, longer.dtype)
    for i in xrange(K):
        convolution[i] = np.dot(longer[i:len(shorter)+i], shorter[::-1])
    return convolution
{% endhighlight %}
 
<div><p>
In Python, this is implemented by <code>numpy.convolve</code> (using <code>valid</code> mode): 
</p></div>

**In [5]:**

{% highlight python %}
a = np.random.normal(0.0, 1.0, 100)
b = np.random.normal(0.0, 1.0, 11)
np.allclose(np.convolve(a, b, mode='valid'), convolve(a, b))
{% endhighlight %}


<div><p>
It is natural to interpret $a\ast b$ as a moving average of $a$, weighted by
$b$. By this view, $a\ast b$ is expected to be of length $N$, i.e. $a\ast b\in
C^N$. That's where boundary effects start to kick in, and the values $(a\ast
b)_n$ for $n\lt M$ and $n\gt N-M$ required to be carefully defined. There is no
one right definition; different applications may point to different strategies.
Usually some padding is involved.
</p></div>

<div><p>
One such common and useful strategy, is to define $a_i=a_0$ for $-M \le i\le 0$
and $a_i=a_{N-1}$ for $N \ge i\ge N+M$. Another, even more common and useful, is
to define $a_i=0$ for $-M \le i\le 0$ or $N \ge i\ge N+M$.
</p></div>

<div><p>
In Python, this too is implemented by <code>numpy.convolve</code> (this time, using
<code>same</code> mode): 
</p></div>

**In [6]:**

{% highlight python %}
np.allclose(np.convolve(a, b, mode='same'),
            convolve(np.pad(a, ((len(b)-1)/2, (len(b)-1)/2), mode='constant', constant_values=0.0), b))
{% endhighlight %}





<div><p>
The same logic can be applied for further undefined elements of $a$, and leads
to a definition of convolution that satisfies $a\ast b\in C^\infty$. Of course,
trivially $(a\ast b)_n=0$ for almost all $n$. The restriction of this definition
to the range of the non-trivial inner-products, so $a\ast b\in C^{N+2M}$, is
what people usually mean by <b>linear convolution</b>.
</p></div>

<div><p>
In Python, this is the <code>full</code> mode of <code>numpy.convolve</code>: 
</p></div>

**In [7]:**

{% highlight python %}
np.allclose(np.convolve(a, b, mode='full'),
            convolve(np.pad(a, (len(b)-1, len(b)-1), mode='constant', constant_values=0.0), b))
{% endhighlight %}


 
<div><p>
It's obviously possible to compute linear convolutions explicitly, without any
paddings: 
</p></div>

**In [8]:**

{% highlight python %}
def direct_linear_convolve(longer, shorter): 
    shorter = shorter[::-1]
    convolution = np.zeros(len(longer)+len(shorter)-1, longer.dtype)
    for i in xrange(1,len(shorter)): 
        convolution[i-1] = np.dot(shorter[-i:], longer[:i])
    for i in xrange(len(longer)-len(shorter)+1):        
        convolution[i+len(shorter)-1] = np.dot(shorter, longer[i:len(shorter)+i])        
    for i in xrange(1, len(shorter)):        
        convolution[i+len(longer)-1] = np.dot(shorter[:-i], longer[len(longer)-len(shorter)+i:])
    return convolution
{% endhighlight %}

**In [9]:**

{% highlight python %}
np.allclose(np.convolve(a, b, mode='full'), direct_linear_convolve(a,b))
{% endhighlight %}



 
<div><p>
This method is not pretty, but at first sight may seem computationally superior:
it performs less floating-point operations, and requires less storage space. The
downside, though, is that due its structure (the first and second loops contain
dot products of varying lengths) parallelizing it usually leads to either branch
divergence or excessive/non-coalesced memory access - both are arch-nemesis of
computational throughput in the common <code>SIMD</code>-based concurrent hardware. 
</p></div>
 
<div><p>
<b>Cross-correlation</b> is another operation over sequences, that is intimately
related to the linear convolution. For $a\in C^{N}$ and $b\in C^{2M+1}$, the
cross-correlation is $(a\star b)_n:=\sum_{k=-M}^{+M}a_{n+k}^\ast b_{M+k}$. In a
sense, the name "crosscorrelational networks" is more appropriate than
"convolutional networks": a "sliding inner-product" is a pretty good description
of what convolutional layers "morally" do. This is nitpicking of course, especially
since in the context of ConvNets (which are usually restricted to real-numbers),
convolutions and cross-correlations are pretty much the same thing: the cross-
correlation of two finite sequences is the convolution of the first with a
reversed version of the second.
</p></div>

<div><p>
An algorithm for linear cross-correlation (or convolution) that simply follows
the definition, has a quadratic time-complexity of $O(NM)$. In Python, it's
given by <code>numpy.correlate</code>.
</p></div>

<div><p>
The generalization of all of the above for the multidimensional setting is
straightforward. For example, if $a\in C^N\times C^N$ and $b\in C^{2M+1}\times
C^{2M+1}$, then their 2-dimensional convolution is $(a\ast b)_{(n,m)}:=\sum_{k_1
=-M}^{+M}\sum_{k_2=-M}^{+M}a_{(n-k_1,m-k_2)}b_{(M+k_1,M+k_2)}$. Most of the
time, the one-dimensional algorithms and results are directly generalizable to
higher-dimensions. So for the sake of notational sanity, most of the following
will focus on one-dimensional sequence, and a separate section will be dedicated
for the implementation details and algorithmic aspects of multidimensional
convolutions. 
</p></div>
 
### 1.3. The Discrete Fourier Transform
<div><p>
There is another, very different, approach for computing convolutions, via the
Discrete Fourier Transform (the DFT). Due to some very nice algorithmic wizardry
(discussed in details later) it leads to asymptotically better
algorithms for computing convolutions.
</p></div>

<div><p>
The DFT maps a finite vector (or a sequence) $a\in C^N$ in the feature space, to
a finite vector $\hat{a}\in C^N$ in the so-called "frequency space", and is
given by $\hat{a}_k=\sum_{n=0}^{N-1}a_ne^{-\frac{2\pi ikn}{N}}$. The transform
is invertible, and its inverse (the IDFT) is given by
$a_k=\frac{1}{N}\sum_{n=0}^{N-1}\hat{a}_ne^{\frac{2\pi ikn}{N}}$. This explains
the term "frequency space": for the $k$-th element in the original sequence, the
$n$-th element of the transformed sequence is the amplitude of a sinusoidal.
</p></div>

<div><p>
Algebraically, the DFT is an orthogonal basis expansion of its argument in
$L^2([0, 2\pi])$ with respect to the trigonometric system $\{e^{\frac{2\pi i
kn}{N}}\}$. Geometrically, the DFT represents each element in the sequence as a
weighted sum of two-dimensional vectors whose angles follow an arithmetic
progression up to a closure of a full circle. In terms of signals, the DFT
correlates the given signal against some sinusoidal basis functions, and the
resulting values are those correlations.
</p></div>

<div><p>
The DFT is denoted by $\mathcal{F}$ and its inverse by $\mathcal{F}^{-1}$. It
worth noting that the DFT is a linear transformation: $\mathcal{F}(a)=W_Na$
where $W_N\in M_{n\times n}(C)$ is the matrix $W_{n,k}=e^{-\frac{i2\pi kn}{N}}$.
A nice way to visualize it can be seen in
<a href="http://www.johndcook.com/blog/2016/01/10/visualizing-the-dft-matrix/">here</a>.
The relevancy of the DFT to our case is due to the <a href="https://en.wikipedia.org/wiki/Convolution_theorem">convolution
theorem</a>: 

$$\mathcal{F}(a\ast b)=\mathcal{F}(a)\cdot\mathcal{F}(b)\Rightarrow a\ast b=\mathcal{F}^{-1}(\mathcal{F}(a)\cdot\mathcal{F}(b))$$
</p></div>

<div><p>
It's easy to understand this theorem in terms of polynomials: any finite
sequence $a\in C^N$ induces a polynomial $P_a$ of degree $N-1$ whose
coefficients are given by the sequence. From the definition, it's immediate that
the convolution $a\ast b\in C^{2N}$ of two sequences $a,b\in C^N$ is given by
the coefficients of the polynomial $P_a\cdot P_b$ (that is, the multiplication
of $P_a$ and $P_b$ in $C[x]$).
</p></div>

<div><p>
The DFT acts on a polynomial by evaluating it on the unit-circle:
$\hat{a}_k=P_a(e^{-\frac{i2\pi k}{N}})$ where $P_a(x)=\sum_{n=0}^{N-1}a_nx^n$.
Thus the DFT takes $N$ numbers (coefficients) and returns $N$ numbers
(evaluations over the unit-circle). When two polynomials are represented by
their evaluations on the same points, their product is given by pointwise
multiplication of their values - and that's what the theorem says. 
</p></div>

**In [10]:**

{% highlight python %}
def straightforward_DFT(sequence):
    N = len(sequence)
    rng = np.arange(N)
    return np.dot(np.exp(-2.0j*np.pi*rng.reshape((N,1))*rng/(N+0.0)), sequence)

def straightforward_IDFT(sequence):
    N = len(sequence)
    rng = np.arange(N)
    return np.dot(np.exp(2.0j*np.pi*rng.reshape((N,1))*rng/(N+0.0)), sequence)/(N+0.0)

def DFT_convolution(a, b):
    pa = np.pad(a, (0, len(b)-1), mode='constant', constant_values=0.0)
    pb = np.pad(b, (0, len(a)-1), mode='constant', constant_values=0.0)    
    return straightforward_IDFT(straightforward_DFT(pa)*straightforward_DFT(pb))
    

a = np.random.normal(0.0, 1.0, 100)
b = np.random.normal(0.0, 1.0, 100)
print np.allclose(DFT_convolution(a,b), np.convolve(a, b, mode='full'))
{% endhighlight %}

 
<div><p>
Theoretically, this can be generalized in several ways. A analogue of
the convolution theorem for functions $G\mapsto C$, where $G$ is a
<a href="https://en.wikipedia.org/wiki/Locally_compact_group">locally-compact group</a>
(of which the scenario above, $Z$, is a special case) is given by
<a href="https://en.wikipedia.org/wiki/Pontryagin_duality">Pontryagin duality</a>,
and there are several generalizations for the functions' range, that
<a href="http://mathworld.wolfram.com/NumberTheoreticTransform.html">for example</a> replaces $C$ with
$Z/pZ$ or other rings (but the characterization of the algebraic structures
that allow fast convolutions is an
<a href="http://mathoverflow.net/questions/10237/does-the-convolution-theorem-apply-to-weaker-algebraic-structures">open question</a>).
</p></div>

<div><p>
Computationally, one issue worthy of some attention, is the computation of the basis elements
$e^{-\frac{i2\pi kn}{N}}$. The code above naively recomputes them on the fly, but
this is clearly sloppy and wasteful. It's much better to precompute those
elements once, and reuse them later.
</p></div>

<div><p>
For small $N$s it actually makes sense to use a $N^2$ matrix, but pretty soon
this approach becomes impractical, and since the group is of order $N$ it's
natural to compute just those $N$ elements (denote them by
$\omega_n=e^{-\frac{i2\pi n}{N}}$). We get
$\hat{a}_k=\sum_{n=0}^{N-1}a_n\omega_n^k$, but an implementation the would
literally follow this formula would be a poor one. Powers of complex numbers are
computationally expensive.
</p></div>

<div><p>
But our elements are not just complex numbers; they are laid on the unit-circle.
So it's possible to precompute just the values
$\alpha_n:=\mathrm{Arg}\omega_n=\frac{2\pi n}{N}$. Then the <code>DFT</code> (or
<code>IDFT</code>) would apply Euler's formula
$\omega_n^k=\cos{(k\alpha_n)}+i\sin{(k\alpha_n)}$ and avoid taking powers. In
<code>OpenCL</code> we could start $N$ work-items, that each would call the
following method with $k$ equals to its local id:
</p></div>


{% highlight c %}
inline void init_roots(unsigned int n, unsigned int N,
                       __local float * const args)
{
   args[n] = -2.0f*M_PI*(n)/(float)(N);
}
{% endhighlight %}


<div><p>
and then the <code>DFT\IDFT</code> methods would look like this:
</p></div>

{% highlight c %}
typedef float2 complex32;
inline complex32 mul(complex32 Z1, complex32 Z2) {
   return (complex32)(Z1.x*Z2.x-Z1.y*Z2.y, Z1.x*Z2.y+Z1.y*Z2.x);
}

inline complex32 dft(unsigned int N, unsigned int k,
                     __local complex32 * const sequence,
                     __local float * const args)
{
   complex32 result = ((complex32)(0.0f, 0.0f));
   for (unsigned int n = 0; n < N; ++n) {
      float arg = args[n]*(float)k;
      result += mul((complex32)(cos(arg), sin(arg)), sequence[n]);
      //result += mul((complex32)(cos(arg), -sin(arg)), sequence[n]); // IDFT
   }
   return result;
   // return result/(float)N; // IDFT
}
{% endhighlight %}

<div><p>
So we can avoid complex powers, but of course, <code>sin</code> and <code>cos</code> are not
exactly a prime a example of quickly computable functions. In many cases
it would be probably much better to use twice the storage and fully precompute
the elements:
</p></div>


{% highlight c %}
inline void init_roots(unsigned int n, unsigned int N,
                       __local float * const roots)
{
   const float omega = -2.0f*M_PI/(float)(N);
   roots[n] = (complex32)(cos(omega*n), sin(omega*n));
}
{% endhighlight %}

<div><p>
and then the <code>DFT/IDFT</code> methods would index the roots using modular
arithmetics $\omega_n^k=\omega_0^{nk}=\omega_0^{nk\text{
mod}(N)}=\omega_{nk\text{ mod}(N)}$:
</p></div>

{% highlight c %}
inline complex32 conj(complex32 Z) { return (complex32)(Z.x, -Z.y); }

inline complex32 dft(unsigned int N, unsigned int k,
                     __local complex32 * const sequence,
                     __local complex32 * const roots)
{
   complex32 result = ((complex32)(0.0f, 0.0f));
   for (unsigned int n = 0; n < N; ++n) {
      result += mul(roots[(n*k)%N], sequence[n]);
      //result += mul(conj(roots[(n*k)%N]), sequence[n]); // IDFT
   }
   return result;
   // return result/(float)N; // IDFT
}
{% endhighlight %}

<div><p>
Finally, there's a third nice possibility - which avoids using modular
arithmetics, and instead uses just multiplications and additions. In some
settings, the could leads to preferable performances (though it may cause some
numerical headaches). The idea is using <a href="https://en.wikipedia.org/wiki/Chebyshev_polynomials">Chebyshev
method</a>, for recursively
computing $\cos(k\alpha_n)$ and $\sin(k\alpha_n)$. e.g. for cosines (that is,
for the real parts), it means that -

$$\cos(k\alpha_n)=2\cos(\alpha_n)\cdot\cos((k-1)\alpha_n)-\cos((k-2)\alpha_n)$$ 
</p></div>

**In [11]:**

{% highlight python %}
N = 14
precomputed_reals = np.cos(2.0*np.pi*np.arange(N)/(N+0.0))
def chebyshev_reals(k):
    res = np.zeros(N)
    res[:2] = [1.0, precomputed_reals[k]]
    for i in xrange(N-2):
        res[i+2] = 2*res[1]*res[i+1]-res[i]
    return res
        
k = 5
print np.allclose(np.cos(2.0*np.pi*np.arange(N)*k/(N+0.0)), chebyshev_reals(k))
{% endhighlight %}

 
<div><p>
A direct computation of the DFT (e.g. by performing $N$ dot-products) takes
$O(N^2)$ steps, which is exactly the same complexity of a direct computation of
the convolution - plus some extra overhead. But as mentioned earlier, there are
clever ways for computing the DFT, by exploiting the group-structure of the
roots of unity, in $O(N\log N)$ steps. By using Fast Fourier Transforms (FFTs)
it's possible to compute a convolution in $O(N\log N)$ as well, which is a
significant asymptotic gain. Soon we'll explore fast algorithms for computing
the DFT. But for the meanwhile and for prototyping, we will use Numpy's
implementation given by <code>numpy.fft.fft</code> and <code>numpy.fft.ifft</code>.
</p></div>

<div><p>
In practice, the constant overhead the transform costs may not always be worth
doing it, and for short sequences a direct computation is sometimes faster
(especially when parallelism is involved). Quadratic-time algorithms for
convolution are useful in practice, so the following will discuss them as well
as quasilinear-time algorithms. Choosing the best approach for a specific
situation can be a tricky business, and it's going to be a recurring theme in
the rest of this post. 
</p></div>
 
## 1.4. Circular Convolution and Spectral Interpolation

<div><p>
Except for truncation and paddings, there is another obvious approach to go from
the definition of convolution for infinite sequences to the finite setting:
given $a\in C^{N}$ and $b\in C^{2M+1}$, embed $a$ within an infinite periodic
sequences with period $N$, and obtain $(a\ast
b)_n:=\sum_{k=-M}^{+M}a_{(n-k)_{\text{mod}N}}b_{M+k}$. This is a <b>circular
convolution</b>. The result, $a\ast b$ is a periodic infinite sequence with period
(at most) $N$.
</p></div>

<div><p>
Obtaining a linear convolution by using a given method that computes a circular
convolution is not hard. The usual "pad and truncate" trick applies: 
</p></div>

**In [12]:**

{% highlight python %}
def circular_convolution(a, b):
    return convolve(np.hstack((a[-len(b)+1:], a)), b)

def linear_by_circular(a, b):    
    return circular_convolution(np.pad(a, (0, len(b)-1), mode='constant', constant_values=0.0), b)
{% endhighlight %}

**In [13]:**

{% highlight python %}
a = np.random.normal(0.0, 1.0, 100)
b = np.random.normal(0.0, 1.0, 21)
print np.allclose(linear_by_circular(a, b), np.convolve(a, b, mode='full'))
{% endhighlight %}



 
<div><p>
The centrality of circular convolutions comes from the fact that computing
convolutions via the convolution-theorem produces circular convolutions: 
</p></div>

**In [14]:**

{% highlight python %}
a = np.random.normal(0.0, 1.0, 13)
b = np.random.normal(0.0, 1.0, 13)
print np.allclose(circular_convolution(a, b), np.fft.ifft(np.fft.fft(a)*np.fft.fft(b)))
{% endhighlight %}

 
<div><p>
What if $a$ and be $b$ are not of the same length? According to the <b>spectral
interpolation</b> theorem, the zero-padding in the space-domain is the same as
interpolation in the frequency domain, hence it's enough to simply zero-pad the
sequences and apply the convolution theorem on the now-equally-sized sequences.
I think it's even clearer by considering the sequences as the coefficients of
two polynomials: certainly each polynomial implicitly has infinite zero
coefficients, and making them explicit doesn't change the evaluations of the
polynomials: 
</p></div>

**In [15]:**

{% highlight python %}
a = np.random.normal(0.0, 1.0, 100)
b = np.random.normal(0.0, 1.0, 21)
b_padded = np.pad(b, (0, len(a)-len(b)), mode='constant', constant_values=0.0)
print np.allclose(circular_convolution(a, b), np.fft.ifft(np.fft.fft(a)*np.fft.fft(b_padded)))
{% endhighlight %}


<div><p>
For cross-correlation, the situation is tricker. Again, in the convenient la-la
land of infinite sequences, we obtain from the basic properties of the Fourier
transform (namely, $\mathcal{F}[f(-x)](\xi)=\mathcal{F}[f(x)](-\xi)$ and
$\mathcal{F}[f(x)^\ast](\xi)=\mathcal{F}[f(x)](-\xi)^\ast$) an analogue of the
convolution theorem for cross-correlation: $(a\star
b)=\mathcal{F}^{-1}(\mathcal{F}(a)^\ast\cdot\mathcal{F}(b))$. But when it comes
to finite sequences, the party is quickly pooped.
</p></div>

<div><p>
To see why, consider the effect of time reversal on the DFT: substituting
$r=N-1-n$ in the definition $\mathcal{F}(a)_k=\sum_{n=0}^{N-1}a_ne^{-\frac{i2\pi
kn}{N}}$ leads to $\mathcal{F}(a)_k=\sum_{r=0}^{N-1}a_re^{-\frac{i2\pi
k(N-1-r)}{N}}=e^{-\frac{i2\pi k(N-1)}{N}}\sum_{r=0}^{N-1}a_re^{\frac{i2\pi
kr}{N}}=e^{\frac{i2\pi k}{N}}\sum_{r=0}^{N-1}a_re^{\frac{i2\pi kr}{N}}$, and
taking the conjugate both inside and outside the summation
($z_1+z_2=(z_1^\ast+z_2^\ast)^\ast$) results with the gloomy conclusion that if
$\overline{a}$ is the time-reversed version of $a$, then
$\mathcal{F}(\overline{a})_k=e^{\frac{i2\pi k}{N}}(\sum_{r=0}^{N-1}a_r^\ast
e^{\frac{-i2\pi kr}{N}})^\ast=e^{\frac{i2\pi k}{N}}\mathcal{F}(a^\ast)^\ast_k$: 
</p></div>

**In [16]:**

{% highlight python %}
a = np.random.normal(0.0, 1.0, 100)*1.0j+np.random.normal(0.0, 1.0, 100)
print np.allclose(np.fft.fft(a[::-1]), np.conj(np.fft.fft(np.conj(a)))*np.exp(2.0j*np.pi*np.arange(len(a))/(len(a)+0.0)))
{% endhighlight %}

 
<div><p>
According to the Shift theorem, a circular temporal shift is the same a linear
phase change, $\mathcal{F}(\{a_{(n-m)\text{mod}N}\})_k=\mathcal{F}(a)_k\cdot
e^{\frac{-2i\pi km}{N}}$. Applied to the result above, it gives
$\mathcal{F}(\{a_{(n-N+1)\text{mod}N}\})_k=\mathcal{F}(a)_k\cdot e^{\frac{-2i\pi
k(N-1)}{N}}=\mathcal{F}(a)_k\cdot e^{-2i\pi k}e^{\frac{2i\pi
k}{N}}=\mathcal{F}(a)_k\cdot e^{\frac{2i\pi k}{N}}$. So the finite cross-
correlational analogue to the convolution theorem is still almost $(a\star
b)=\mathcal{F}^{-1}(\mathcal{F}(a)^\ast\cdot\mathcal{F}(b))$, only that the
pointwise multiplication involves a shifted version of $\mathcal{F}(a)$ by an
offset of $N-1$.
</p></div>

<div><p>
One choice that leaves us, is to simply use the cross-correlation theorem "as
is" and shift the result: 
</p></div>

**In [17]:**

{% highlight python %}
a = np.random.normal(0.0, 1.0, 100)+1.0j*np.random.normal(0.0, 1.0, 100)
b = np.random.normal(0.0, 1.0, 21)+1.0j*np.random.normal(0.0, 1.0, 21)
padded_b = np.pad(b, (0, len(a)-len(b)), mode='constant', constant_values=0.0)
shifted = np.fft.ifft(np.fft.fft(a)*np.conj(np.fft.fft(padded_b)))
np.allclose(circular_convolution(a, np.conj(b)[::-1]), np.roll(shifted, len(b)-1))
{% endhighlight %}


 
<div><p>
That could sometimes lead to troubles, as in the many situations in which is best
to relay on the linearity of the DFT to perform accumulations and weightings in
the frequency domain. Many times (and convolutional networks are not an
exception), this brings significant computational benefits. But this annoying
shifting can sometimes require annoyinger bookkeeping.
</p></div>

<div><p>
Another choice, is to explicitly apply the Shift theorem: 
</p></div>

**In [19]:**

{% highlight python %}
def time_shift(transformed, offset):
    N = len(transformed)
    return np.exp(-2.0j*np.pi*np.arange(N)*offset/(N+0.0))*transformed

a = np.random.randint(0, 100, 9)
b = np.random.randint(0, 100, 5)
c = np.random.randint(0, 100, 9)
d = np.random.randint(0, 100, 5)

t_a = np.fft.fft(a)
t_b = np.fft.fft(np.pad(b, (0, len(a)-len(b)), mode='constant', constant_values=0.0))
t_c = np.fft.fft(c)
t_d = np.fft.fft(np.pad(d, (0, len(c)-len(d)), mode='constant', constant_values=0.0))

shift = np.fft.ifft(time_shift(np.conj(t_a), len(a)-1)*t_b +
                    time_shift(np.conj(t_c), len(c)-1)*t_d)
print 'Expected:        ', circular_convolution(a[::-1], b) + circular_convolution(c[::-1], d)
print 'Result (shift):  ', np.round(np.real(shift)).astype(np.int64)
{% endhighlight %}

**Out [19]:**

    Expected:         [31626 35675 34911 30970 34250 27120 32454 38685 32808]
    Result (shift):   [31626 35675 34911 30970 34250 27120 32454 38685 32808]

 
<div><p>
Here's a summary-in-code for the discussion up to this point: 
</p></div>

**In [20]:**

{% highlight python %}
class Spatial(object):
    @staticmethod
    def convolve(a, b):
        return np.convolve(a, b, mode='valid')
    
    @staticmethod
    def correlate(a, b):
        return np.convolve(a, np.conj(b)[::-1], mode='valid')
    
    @staticmethod
    def transform_linear(a, b):
        return np.pad(a, (len(b)-1, len(b)-1), mode='constant', constant_values=0.0), b
    
    @staticmethod
    def transform_circular(a, b):
        return np.hstack((a[-len(b)+1:], a)), b
    
    
class Spectral(object):
    @staticmethod
    def convolve(a, b):
        assert(len(a)==len(b))
        return np.fft.ifft(a*b)
    
    @staticmethod
    def correlate_spatial_roll(a, b, K):
        assert(len(a)==len(b))
        return np.roll(np.fft.ifft(a*np.conj(b)), K-1)

    @staticmethod
    def correlate_spectral_roll(a, b, K):
        assert(len(a)==len(b))        
        shifted_b = np.exp(-2.0j*np.pi*np.arange(len(b))*(K-1)/(len(b)+0.0))*np.conj(b)
        return np.fft.ifft(a*shifted_b)
    
    @staticmethod
    def transform_linear(a, b):        
        K = len(b)
        a_spectral_padding = np.pad(a, (0, len(b)-1), mode='constant', constant_values=0.0)
        b_spectral_padding = np.pad(b, (0, len(a)-1), mode='constant', constant_values=0.0)
        return np.fft.fft(a_spectral_padding), np.fft.fft(b_spectral_padding), K

    @staticmethod
    def transform_circular(a, b):
        K = len(b)
        return np.fft.fft(a), np.fft.fft(np.pad(b, (0, len(a)-len(b)), mode='constant', constant_values=0.0)), K
{% endhighlight %}

**In [21]:**

{% highlight python %}
a = np.random.normal(0.0, 1.0, 100)+1.0j*np.random.normal(0.0, 1.0, 100)
b = np.random.normal(0.0, 1.0, 21)+1.0j*np.random.normal(0.0, 1.0, 21)

spetial_a, spetial_b = Spatial.transform_linear(a, b)
spectral_a, spectral_b, K = Spectral.transform_linear(a, b)
assert(np.allclose(Spatial.convolve(spetial_a, spetial_b), Spectral.convolve(spectral_a, spectral_b)))
assert(np.allclose(Spatial.correlate(spetial_a, spetial_b), Spectral.correlate_spatial_roll(spectral_a, spectral_b, K)))
assert(np.allclose(Spatial.correlate(spetial_a, spetial_b), Spectral.correlate_spectral_roll(spectral_a, spectral_b, K)))

spetial_a, spetial_b = Spatial.transform_circular(a, b)
spectral_a, spectral_b, K = Spectral.transform_circular(a, b)
assert(np.allclose(Spatial.convolve(spetial_a, spetial_b), Spectral.convolve(spectral_a, spectral_b)))
assert(np.allclose(Spatial.correlate(spetial_a, spetial_b), Spectral.correlate_spatial_roll(spectral_a, spectral_b, K)))
assert(np.allclose(Spatial.correlate(spetial_a, spetial_b), Spectral.correlate_spectral_roll(spectral_a, spectral_b, K)))
{% endhighlight %}


