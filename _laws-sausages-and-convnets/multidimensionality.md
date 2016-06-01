---
layout: laws-sausages-and-convnets
title: "5. Multidimensionality"
---

<div><p>
So far, only 1-dimensional convolutions were considered. Most of the insights
regarding one-dimensional convolutions can be applied as-is for multidimensional
convolutions (which are the common case in practice). Yet, there are some
additional issues that are associated with increasing the dimension.
</p></div>

<div><p>
They come in two flavours: technical and algorithmic. The technical issues this
section addresses are image memory objects and shared-memory bank conflicts,
and the algorithmic issues revolve around ways to compute multidimensional
convolutions via 1-dimensional convolutions, and include separable filters,
the row-column method, and the Helix transform. For readability, I shall
focus on the 2-dimensional case.
</p></div>

## 5.1. Separability
<div><p>
As for algorithmic issues for multidimensional layers, let's start with
<a href="https://en.wikipedia.org/wiki/Separable_filter"></b>separable filters</b></a>. Those
involve multidimensional convolutions that can be done by performing
1-dimensional convolutions separably on each dimension. On the 2-dimensional
case, it's simply means that the template-matrix is of rank 1. Now, very much
depending on the context in which the convolutional layer in used in the neural
network, it could be the case that it makes sense to learn such filters (for
example, it's possible to learn an edge-detector for input images that way).
</p></div>

<div><p>
This is a design issue, and it's implemented by maintaining 2 weight-vectors pf
length $(2M+1)$ instead of 1 weight-matrix of order $(2M+1)^2$. The outer-
product of those vectors will be used as the theoretical template. In practice,
though, such layers can be implemented as 2-successive 1-dimensional layers: the
first operates on rows and the second operates on cols. Such a separation makes
backpropagation simpler.
</p></div>

## 5.2. The Row-Column Method and The Helix Transform
<div><p>
Let $a\in C^{N_1\times N_2}$, and denote $\vec{a}_n$ the $n$-th row of $a$
(which is a vector of length $N_2$). Then by definition it's 2-dimensional
discrete Fourier transform is -

$$\mathcal{F}[a]_{n_1,n_2}:=\sum_{k_1=0}^{N_1-1}\sum_{k_2=0}^{N_2-1}a_{k_1,k_2}\omega_{N_1}^{n_1k_1}\omega_{N_2}^{n_2k_2}=\sum_{k_1=0}^{N_1-1}\omega_{N_1}^{n_1k_1}(\sum_{k_2=0}^{N_2-1}a_{k_1,k_2}\omega_{N_2}^{n_2k_2})=\sum_{k_1=0}^{N_1-1}\mathcal{F}[\vec{a}_{k_1}]_{k_2})\omega_{N_1}^{n_1k_1}$$
</p></div>

<div><p>
So instead of a 2-dimensional DFT, we can perform $N_1$ 1-dimensional DFTs
over $a$'s rows, and then another $N_2$ 1-dimensional DFTs over the columns of
the transformed matrix.
</p></div>

<div><p>
This is the "row-column method". Apart from the much-appreciated option to reuse
efficient 1-dimensional code for multidimensional problems, this method is
remarkably efficient asymptotically: by using FFTs we can perform 1-dimensional
DFT in $O(N\log{N})$ steps instead of $O(N^2)$ steps. So the row-column method
allow us use the same algorithm to perform 2-dimensional DFT in $O(N^2\log{N})$
steps instead of $O(N^4)$ steps. If follow that this method makes 2-dimensional
FFTs more relatively-efficient than 1-dimensional FFTs. 
</p></div>

**In [1]:**

{% highlight python %}
def dft_rows(matrix):
    for row in xrange(matrix.shape[0]):
        matrix[row, :] = np.fft.fft(matrix[row, :])
    return matrix

def row_column_DFT(matrix):    
    return dft_rows(dft_rows(matrix.copy()).T).T
{% endhighlight %}

**In [2]:**

{% highlight python %}
matrix = np.random.normal(0.0, 1.0, (128, 64)).astype(np.complex128)
assert(np.allclose(np.fft.fft2(matrix), row_column_DFT(matrix)))
{% endhighlight %}

 
<div><p>
Not surprisingly, again the complications of writing an efficient parallelization
of the row-column method is essentially due to the "transpose" operation between
the rows-phase and the columns-phase.
</p></div>
 
<div><p>
A very similar method for translating a multidimensional convolution into a 
1-dimensional convolution is <a href="https://www.ualberta.ca/~mostafan/Files/Papers/md_convolution_TLE2009.pdf">The Helix Transform</a>.
This is a fancy name for treating the matrices involved as vectors with
column-major layout.
</p></div>

**In [3]:**

{% highlight python %}
X = np.random.normal(0.0, 1.0, (3, 2))
Y = np.random.normal(0.0, 1.0, (2, 2))

X_tag = np.zeros((X.shape[0]+Y.shape[0]-1, X.shape[1]+Y.shape[1]-1))
X_tag[:X.shape[0], :X.shape[1]]= X

Y_tag = np.zeros((X.shape[0]+Y.shape[0]-1, X.shape[1]+Y.shape[1]-1))
Y_tag[:Y.shape[0], :Y.shape[1]]= Y

Z_tag = np.convolve(X_tag.T.flatten(), Y_tag.T.flatten(), mode='full')[:len(X_tag.flatten())]
Z_tag = Z_tag.reshape((X_tag.shape[1], X_tag.shape[0])).T

Z_traget = scipy.signal.convolve2d(X, Y, mode='full')
assert(np.allclose(Z_traget, Z_tag))
{% endhighlight %}


## 5.3. Image Memory Objects
<div><p>
Previously, the input for the kernel was placed inside a <code>__global</code> memory
buffer, and cached into the local memory by the <code>cache_tile</code> method. But it
could've been better had we've used <b>image memory objects</b> instead. Those are
opaque memory objects that can be accesses via coordinates much like global
multidimensional arrays.They are restricted to 1, 2 or 3 dimensions. This is
hardly a problem for most current use-cases of convolutional networks, but it is
a restriction nonetheless.
</p></div>

<div><p>
The benefits they provide include optimized caching of multidimensional data and
automatic handling of out-of-bounds reads. Their opaqueness allows the runtime
to do some neat things, such as directly loading 2-dimensional arrays from the
host into a <a href="https://en.wikipedia.org/wiki/Z-order_curve">Morton-order</a> layout
on the device, to improve data locality.
</p></div>

<div><p>
A 2-dimensional image-based code analogues to the 1-dimensional tiled
convolution from before, would like that:
</p></div>

{% highlight c %}
inline float convolve2D_middle(__read_only image2d_t inputs,
                               __constant float const * const weights,
                               unsigned int x, unsigned int y,
                               unsigned int M1, unsigned int M2,
                               sampler_t sampler)
{
   float result = 0.0f;
   const unsigned int WIDTH = 2*M1+1;
   const unsigned int HEIGHT = 2*M1+1;
   for (unsigned int dy = 0; dy < HEIGHT; ++dy) {
      for (unsigned int dx = 0; dx < WIDTH; ++dx) {
         result += weights[dx+dy*WIDTH]*read_imagef(inputs, sampler,
                                                   (int2)(x+M1-dx, y+M2-dy)).x;
      }
   }
   return result;
}


__kernel void convolve2D(__read_only image2d_t inputs,
                         __constant float const * const weights,
                         __write_only image2d_t outputs,
                         unsigned int M1, unsigned int M2,
                         sampler_t sampler)
{
   unsigned int idx = get_global_id(0);
   unsigned int idy = get_global_id(1);
   float result = convolve2D_middle(inputs, weights, idx, idy, M1, M2, sampler);
   write_imagef(outputs, (int2)(idx, idy), (float4)(result, 0.0f, 0.0f, 0.0f));
}
{% endhighlight %}

<div><p>
Note that time no <code>__local</code> tiling are used, since image memory objects
reside in the <code>__global</code> memory. The caching and opaque (hopefully locality-
friendly) layout supposed to make for it. And there are no special handling for
boundaries: the <code>sampler</code> takes care of it. The host-side code should create
the sampler using a code similar to this:
</p></div>

{% highlight c %}
cl_sampler_properties sp[] = {CL_SAMPLER_NORMALIZED_COORDS,
                              CL_FALSE,
                              CL_SAMPLER_ADDRESSING_MODE,
                              CL_ADDRESS_CLAMP_TO_EDGE,
                              CL_SAMPLER_FILTER_MODE,
                              CL_FILTER_NEAREST, 0};
cl_sampler sampler = clCreateSamplerWithProperties(context, sp, &error);
{% endhighlight %}

<div><p>
where the parameter <code>CL_ADDRESS_CLAMP_TO_EDGE</code> determines the boundaries
policy (when using an older version than <code>OpenCL 2.0</code>, use  the deprecated
<code>clCreateSampler</code> instead of <code>clCreateSamplerWithProperties</code>). Note the
usage of <code>float4</code>, which is there because the interface of image-memory
objects always assumes 4-channels - even when just 1 is used, as in our case
here. 
</p></div>
 
## 5.4. Shared-Memory Bank Conflicts
<div><p>
If instead a <code>__local</code> memory buffer is used to cache tiles in a
multidimensional convolution, then another micro-optimization may become
relevant: <b>shared-memory bank conflicts</b>. On most architectures, sequential
words from the local memory are dealt cyclically by sequential memory banks.
This means that upon accessing the local memory, the different work-items within
a frontwaves should address words that belong to different banks or otherwise
incur latencies.
</p></div>

<div><p>
Many times bank conflicts are not a big-deal and not worth dealing with,
especially when there are many active work-groups, so some can work while others
wait (this is called "latency hiding"). At other times, the problem can be
prevented  by a good choice of memory layouts (e.g. Morton-order). Anyway, in
the case of tiled multidimensional convolutions it is always very simple to
solve by just making sure that the tile width (on which we can control) is not
evenly divisible by the number of shared memory banks.
</p></div>
