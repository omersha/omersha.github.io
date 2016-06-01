---
layout: laws-sausages-and-convnets
title: "4. Overlap Methods"
---
 
## 4.1. Parallelized Filters
<div><p>
In essence, the algorithm of a filter with a template $W$ and input $I$ is simply
a linear convolution $I\ast W$. In the context of convolutional layers, this is
the structure of both their forward algorithm and the backward algorithm for
$\frac{dE}{dI}$. As we've seen, there are two general strategies for designing
such an algorithm: spatial (based on dot-products) and spectral (based on
the <code>DFT</code>).
</p></div>

<div><p>
Asymptotically, the spectral strategy is better due to Fast
Fourier Algorithms. But it comes with an overhead that could
give an advantage to the spatial strategy for small problems. Either way,
though, the computational throughput of this calculation can be significantly
improved by parallelism.
</p></div>

<div><p>
Implementing parallel algorithms is never trivial, and despite its simplicity -
even direct convolution is not an exception. It's an interesting case-study, that
demonstrates some very useful and general optimization patterns (see <a href="http://impact.crhc.illinois.edu/shared/papers/optimization2012.pdf">Stratton
at el</a>).
</p></div>

<div><p>
For the purpose of this section, I'll assume a one-dimensional convolutional
layer that maintains a set of weights $W\in R^{2M+1}$ and whose inputs are given
by $I\in R^N$ with $N\gg 2M+1$. Thus $(W\ast I)\in R^N$. None of those
assumptions is restrictive, and they are imposed here just to make the
discussion and code a bit simpler. Later sections will explicitly deal with
multidimensional layers and with the implications of working over the reals.
Direct spatial algorithms have time-complexity of $O(NM)$, and direct spectral
algorithm have time-complexity of $O(N\log{N})$.
</p></div>

<div><p>
Let's start by spelling out a method that computes the $M$-th element in a
linear convolution of two sequences of length $2M+1$. This method would be a
building block in the following discussion, and the following implementation -
while simple - can be actually sensible in practice in some scenarios. Later
more sophisticated ideas will be introduced.
</p></div>

{% highlight c %}
// Both sequences are of length 2M+1
inline float convolve_middle(__local float const * const sequence1,
                             __global float const * const sequence2,
                             unsigned int M)
{
   float result = (sequence1[M]*sequence2[M]);
   for (unsigned int i = 0; i < M; ++i) {
      result += (sequence1[i]*sequence2[2*M-i]) + (sequence1[i+M+1]*sequence2[M-i-1]);
   }
   return result;
}
{% endhighlight %}

<div><p>
We're ready to think about filters. The first parallelization idea I will
immediately dismiss, is to assign a work-item (aka thread) to each input-
element. The thread will multiple the element with all the weights, and
accumulate the results in the corresponding output elements. This is a
algorithmic pattern known as <b>scatter</b>, and it's a bad choice here since the
different writes will have to be synchronized, and the impact on the
performances is likely to be so dramatic, that calculating the convolution by
hand, using pen and paper, might become a viable alternative.
</p></div>

<div><p>
This leads to the first optimization pattern involved here, "<b>Scatter to Gather
Conversion</b>". Assigning a work-item to each output-element doesn't require
synchronization at all.
</p></div>

{% highlight c %}
__kernel void convolve(__global float const * const inputs,
                       __global float const * const weights,
                       __global float * const outputs,
                       unsigned int M)
{
   unsigned int idx = get_global_id(0);
   outputs[idx] = convolve_middle(inputs+idx, weights, M);
}
{% endhighlight %}

<div><p>
The code is correct, but the way it accesses memory is a complete mess. Each
work-item reads from the global memory $2M+1$ for the inputs, and another $2M+1$
for the weights. As a rule of thumb, the performance of a good implementation
for a parallel algorithm is determined its utilization of the memory bandwidth -
and the code above has lousy throughput.
</p></div>

<div><p>
This can be fixed by employing a common optimization pattern, we've already used
for transpositions: "<b>Tiling</b>". The idea is to divide the inputs into small
chunks ("tiles"), and assign each chunk a work-group (known as "thread-blocks"
in CUDA-lingo). All the work-items within the same work-group will work on the
same input elements - but first, they will copy those elements from the global
memory into the local memory in a coalesced manner. The local memory is shared
between items of a work-group, and is much faster than the global memory. For
spectral algorithms tiling could have an additional benefits, because it would
allow us to rely solely on an optimized FFT algorithm for small and constrained
sizes.
</p></div>

<div><p>
Methods that implement linear filters by decomposing the large-convolution into
many small-convolutions are known by the signal-processing folks as "overlap
methods", and they come in two variants: <b>overlap-add</b> partitions an input
sequence into disjoint blocks and produces overlapping slices of the output, and
<b>overlap-discard</b> works on overlapping slices of the input sequence, and
produces disjoint slices of the output.
</p></div>

<div><p>
The motivation for introducing them here is parallelism, but another thing
they're good for is reduction of latency. Those methods can produce the result
incrementally while the computation is getting done. This is usually not a major
factor for ConvNets where the entire convolution is often required for the next
steps anyway, and when the major computational bottleneck is in the training
stage where the main optimization objective is throughput maximization, and not
latency minimization.
</p></div>

<div><p>
With these methods, it's possible to use a given implementation for convolving
two fixed-sized small sequences (possibly highly optimized), for convolving
arbitrary sequences. 
</p></div>
 
## 4.2. Overlap-Discard
<div><p>
Let's start with overlap-discard. This methods works on overlapping blocks of
the input, and produces disjoint blocks of the output. Let $I\in C^N$ and $W\in
C^M$ ($M\ll N$). The algorithm first slices $I$ into small overlapping blocks
$I_k$ of length $L+M$ ($M\le L\ll N$): 
</p></div>

**In [1]:**

{% highlight python %}
def overlap_discard_slice(sequence, M, L, k):
    return sequence[k*L:(k+1)*L+2*M]
{% endhighlight %}

**In [2]:**

{% highlight python %}
print overlap_discard_slice(np.arange(15), 2, 5, 0)
print overlap_discard_slice(np.arange(15), 2, 5, 1)
print overlap_discard_slice(np.arange(15), 2, 5, 2)
{% endhighlight %}

**Out [2]:**
<pre>
    [0 1 2 3 4 5 6 7 8]
    [ 5  6  7  8  9 10 11 12 13]
    [10 11 12 13 14]
</pre>
 
<div><p>
Then, for each $I_k$ it computes a corresponding block $O_k$ of $I\ast W$ (where
the $O_k$s consist a disjoint partitioning of $I\ast W$). The algorithm relies
on the fact that the minimal convolution $I_k\ast W$ of each block with the
weights (that is, when no boundary-effects are in play) is of length
$(L+M)-M=L$, and their concatenation is $I\ast W$.
</p></div>

<div><p>
The "discard" in its name comes from the common approach of computing $I_k\ast
W$ by using the DFT, which makes the asymptotic computational complexity of the
algorithm $O(\frac{N}{L}L\log{L})=O(N\log{L})$ instead of
$O(\frac{N}{L}L^2)=O(NL)$. The DFT produces a circular convolution of length
$L+M$ whose first $M$ elements are meaningless by-products of the "wrap-around"
effect, and are discarded. Note that a spectral overlap-discard algorithm is
asymptotically better than a direct spectral convolution, while a spatial
overlap-discard is asymptotically equivalent to a spatial direct convolution.
</p></div>

<div><p>
By it's nature, the algorithm produces a minimal convolution ("valid" in
<code>numpy</code> terms). But as discussed earlier, obtaining a full linear
convolution from this is an easy task, accomplished by zero-padding $I$. 
</p></div>

**In [3]:**

{% highlight python %}
def overlap_discard(I, W, L):
    return np.hstack([circular_convolution(I[k*L:(k+1)*L+len(W)-1], W)[len(W)-1:] for k in xrange(len(I)/L+1)])
{% endhighlight %}

**In [4]:**

{% highlight python %}
L = 21
I = np.arange(100)
W = np.arange(13)
assert(np.allclose(np.convolve(I, W, mode='valid'), overlap_discard(I, W, L)))
{% endhighlight %}

<div><p>
Time to move from prototyping to actual parallel implementation. The overlap-
discard algorithm is easy to parallelize. Each of the $I_k$ blocks will be
associated with a work-group whose work-items will produce $I_k\ast W$. The
<b>gather</b> algorithmic pattern leads naturally to work-groups of $L$
work-items.
</p></div>

<div><p>
All the work-items in a work-group operate on the same data, so it could be
fetched from the global memory in a coalesced manner and cached into the local-
memory. The $L$ work-items would have to load $L+M-1$ data-elements for the
convolution they compute. This will require a total of
$\frac{N}{L}(L+M)=N(1+\frac{M-1}{L})\lt 2N$ global memory reads and $N$ global
memory writes. The $M-1$ "extra" elements are called "halo".
</p></div>

<div><p>
This suggests that the value of $L$ should be as large as possible. There are
several constrains and relevant considerations, though. First, the size of $L$
is limited by the hardware. For GPUs its maximal size is typically in the
~500-2000 range. For CPUs it could get considerably larger (~8000-10000), but
then again, CPUs are much less sensitive to memory access. Another limiting
factor is that both GPUs and CPUs can benefit greatly when the value of $L$ is
divisible by their <code>SIMD</code>-width. And finally, if the circular convolution
step is performed by using a FFT algorithm, it's often best to choose $L$ for
which the algorithm is optimal.
</p></div>

<div><p>
Loading the tile is not hard, but not trivial either. The tile size is larger
than the number of work-items, so it's impossible to naively divide the loading
processing between the work-items. In our setting of convolutional networks,
it's almost always best to compute each sub-convolution via the DFT. As we've
seen, this requires zero-padding of size $M-1$ (so the total tile size is
$L+2(M-1)$). The implementation presented here will use spatial convolutions,
which are useful in practice too. They require tiles of size $L+M-1$ for circular convolutions.
</p></div>

<div><p>
Those loads shouldn't be assigned arbitrarily to work-items, since it's best
that each wrap/wavefront will load sequential elements.  Instead, there will be
$L$ items whose loading will be divided between the $L$ work-items, and from
those there will be $M-1$ work-items that will additionally load the remaining
$M-1$ elements.
</p></div>


{% highlight c %}
inline void cache_spatial_circular_tile(__global float const * const inputs,
                                        __local float * const tile,
                                         unsigned int tile_start,
                                         unsigned int local_idx,
                                         unsigned int M)
{
   if (local_idx < M-1) {
      unsigned int local_index = 2*local_idx;
      unsigned int global_index = tile_start + local_index;
      tile[local_index] = inputs[global_index]; // (global_index<N)?...:0.0f;
      tile[local_index+1] = inputs[global_index+1]; //
(global_index+1<N)?...:0.0f;
   } else {
      unsigned int local_index = M+local_idx-1;
      unsigned int global_index = tile_start + local_index;
      tile[M-1+local_idx] = inputs[global_index]; // (global_index<N)?...:0.0f;
   }
}
{% endhighlight %}

<div><p>
All the global boundary checks (hinted in the comments) are ugly and somewhat
wasteful. The best practice is to write a customized method for the last work-
group with boundary checks (either way branch divergence is not an issue here).
</p></div>

<div><p>
Once the tile is loaded, the convolution can be computed. Since we're working
spatially, it's silly to actually compute a  circular convolution and discard some
of the resulting elements. Instead, it's obviously better to compute just the
elements we're going to use, so we'll just compute a minimal convolution:
</p></div>

{% highlight c %}
inline float convolution_element(__local float const * const sequence1,
                                 __constant float const * const sequence2,
                                 unsigned int M)
{
   float result = 0.0f;
   for (unsigned int i = 0; i < M; ++i) {
      result += (sequence1[i]*sequence2[M-i-1]);
   }
   return result;
}


__kernel void convolve(__local float * const tile,
                       __global float const * const inputs,
                       __constant float const * const weights,
                       __global float * const outputs,
                       unsigned int M,
                       unsigned int N)
{
   unsigned int idx = get_global_id(0);
   unsigned int local_idx = get_local_id(0);
   unsigned int group_idx = get_group_id(0);
   unsigned int L = get_local_size(0);

   cache_spatial_circular_tile(inputs, tile, group_idx*L, local_idx, M, N);
   barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

   if (idx < N) {
      outputs[idx] = convolution_element(tile+local_idx, weights, M);
   }
}
{% endhighlight %}

<div><p>
The code above should be launched with one work-item per output element (i.e.
$N-M+1$ work-items). Note that now the <code>weights</code> are in the constant memory,
which is often faster than the global memory (due to a dedicated small cache).
That's a bit faster than caching them in the local-memory as well, since the
global cache can be shared between work-groups.
</p></div>

<div><p>
And a final word regarding coalescence and alignment. The work-items within a
work-group are divided into wraps/wavefronts. To maximize the throughput,
different reads from work-items within the same wavefront should share the DRAM
bus bandwidth. This basically means that they should read/write consecutive
words, and that the entire chunk of memory read by a wavefront need to be
aligned.
</p></div>

<div><p>
The exact alignment requirements change with the specific architecture in use. A
64-byte alignment requirement is typical. One possible approach is to zero-pad
the inputs when necessary, but this is not always applicable. As an alternative
approach, the kernel can deal with this explicitly. So for example the first
work-group could compute just the "reminder" (less than $M$) elements, so that
the rest of the work group could work as-is on aligned data. 
</p></div>
 
## 4.3. Overlap-Add

<div><p>
The overlap-add method is kinda dual to overlap-discard. Instead of working on
overlapping input-blocks and producing disjoint output-blocks, it works on
disjoint input-blocks and produces a overlapping output-blocks that can't be
simply concatenated. Instead, they are accumulated.
</p></div>

<div><p>
The rationale here is even simpler. As before, let $I\in C^N$ and $W\in C^M$
($M\ll N$). Now, if we denote by $I_k\in C^N$ a copy of $I$ with all the
elements before the $kL$-th element or after the $(k+1)L$ element zeroed, we get
$\sum{I_k}=I$, and $I\ast W=(\sum_k{I_k})\ast W=\sum_k{(I_k\ast W)}$. Here the
sub-convolutions are regular linear convolutions, so $I_k\ast W\in C^{L+M-1}$.
</p></div>

<div><p>
So processing a block requires working on $L$ input elements, and results with
$L+M-1$ output-elements. The last $M-1$ elements of the $k$-th block are
required to be added to the first $M-1$ elements of the $(k+1)$-th block. Hence
the name of this algorithm. Asymptotically, it's exactly the same as the
overlap-discard algorithm.
</p></div>

<div><p>
The algorithm produces a linear convolution (unlike the minimal convolution
produced by overlap-discard): 
</p></div>

**In [5]:**

{% highlight python %}
def overlap_add(I, W, L):
    convolution = np.zeros(len(I)+len(W)-1)    
    for k in xrange(int(np.ceil(len(I)/L)+1)):
        convolution[k*L:(k+1)*L+len(W)-1] += convolve(np.pad(I[k*L:(k+1)*L], (len(W)-1, len(W)-1), mode='constant', constant_values=0.0), W)
    return convolution
{% endhighlight %}

**In [6]:**

{% highlight python %}
L = 21
I = np.arange(100)
W = np.arange(13)
assert(np.allclose(np.convolve(I, W, mode='full'), overlap_add(I, W, L)))
{% endhighlight %}

 
<div><p>
The first step in the Implementation is tiling. In the spatial version of the
algorithm the tile sizes are $L+2(M-1)$, where $2(M-1)$ of their elements are
simply zeros (the required padding for spatial linear convolutions). On the
other, for employing a gathering strategy, each work-group should have $L+M-1$
work-items. So now we're in a situation with less loads than work-items. In
many ways that's easier: $L$ work-items will do the caching, and the remaining
$M-1$ work-items will do the padding:
</p></div>

{% highlight c %}
inline void cache_spatial_linear_tile(__global float const * const inputs,
                                      __local float * const tile,
                                      unsigned int tile_start,
                                      unsigned int local_idx,
                                      unsigned int M,
                                      unsigned int L,
                                      unsigned int N)
{
   if (local_idx < L) {
      unsigned int global_index = tile_start + local_idx;
      tile[local_idx+M-1] = inputs[global_index]; // (global_index<N)?...:0.0f;
   } else {
      tile[local_idx-L] = 0.0f;
      tile[local_idx+M-1] = 0.0f;
   }
}
{% endhighlight %}

<div><p>
The kernel itself is seemingly very similar to overlap-discard's. Efficiency-
wise, the differences between overlap-add and overlap-discard are minor and in
practice their performances are about the same. But with respect to our main
motivation, parallelism, they're very different. As we have seen, paralleling
overlap-discard is (relatively) easy since it produces independent blocks. The
situation with overlap-add is different: the resulting blocks are overlapping
and need to be accumulated. These dependencies complicate the reduction step.
</p></div>

<div><p>
So why should we bother with overlap-add? Can't we use overlap-discard
exclusively and avoid synchronization issues? Why do bad things happen to good
people? Can't we all just get along? Ok, enough with the rhetorical wishful
thinking. Overlap-discard is great in many situations, but sometimes - like, in
the specific context of backpropagation of convolutional layers - it introduces
some acute problems that will be discussed in the next section, "Convolution in
Parts". The bottom line is that the parallelization of the overlap-add algorithm
must be dealt with.
</p></div>

<div><p>
The brute-force solution would be to synchronize work-groups. This could be
potentially be implemented in a non-completely-horrible way, since the
congestion is low: each of the $O(N)$ output-values is modified by at most 2
work-items. The simplest synchronization method here is via atomicity (either
atomic addition or atomic incrementation), but co-processors tend to suffer from
an inefficient coherence protocols between different cores which could destroy
the throughput despite the low congestion (and besides, <code>OpenCL</code> doesn't
provide atomic operations for floating points anyway).
</p></div>

<div><p>
Direct synchronization via locks could be possible, but it's clumsy. A much
(much) better approach for exploiting the fact that each output-values is
modified by at most 2 work-items, is to simply write a 2-step (or generally
$K$-step, with $K\gt 2$) algorithm that makes sure that at each step all the
active work-groups never effects the same outputs. As long as no two adjacent
blocks are executed in the same step, we're golden. For example ($N=10$ and
$K=3$): 
</p></div>

**In [7]:**

{% highlight python %}
def steps_plan(blocks, steps):
    for step in xrange(steps):
        print 'Step %d:'%step, range(step, blocks, steps)
steps_plan(10, 3)
{% endhighlight %}

**Out [7]:**
<pre>
    Step 0: [0, 3, 6, 9]
    Step 1: [1, 4, 7]
    Step 2: [2, 5, 8]
</pre>
 
<div><p>
The choice of$K$ is a micro-optimization, and mostly the choice $K=2$ will be
near-optimal (or just optimal).
</p></div>

<div><p>
Unlike overlap-discard, the implementation of parallel overlap-add on an
heterogeneous machine puts some meaningful logic (not just boilerplate code) on
the host side. A sensible implementation is to launch $K$ kernels using an in-
order queue. So if the kernel is defined like this:
</p></div>

{% highlight c %}
__kernel void convolve(unsigned int step, unsigned int steps, ...)
{
// ..
unsigned int tile_index = get_group_id(0)*steps + step;

// ..
}

{% endhighlight %}

<div><p>
then in the host, the code would have the following form:
</p></div>

{% highlight c %}
// ...
std::vector<cl_kernel> kernels(steps);

// ...
for (unsigned int step = 0; step < steps; ++step) {
     kernels[step] = clCreateKernel(/*...*/);
     clSetKernelArg(kernels[step], 0, sizeof(step), &step);
     clSetKernelArg(kernels[step], 1, sizeof(steps), &steps);
     // ...
     clEnqueueNDRangeKernel(queue, kernels[step], /*...*/);
}
// ...
{% endhighlight %}

<div><p>
An alternative approach, which could lead to better performance but is usually
overly convoluted (no pun intended), is to use out-of-order queues on manage
dependencies with events. The idea is to partition the work-groups of each step
into several disjoint subsets of work-groups (each subset should be large enough
to allow full occupancy of the concurrency capacity). Then the data associated
with the work-groups of each subgroup could be loaded into device-buffers using
out of order queues, and the corresponding kernel would wait on the writing
events.
</p></div>

|                    | Spatial Overlap-Discard             | Spatial Overlap-Add
|:-------------------|:-----------------------------------:|:-----------------------------------:|
| Work Groups        | $\lceil\frac{N+2(M-1)}{L}\rceil$    | $\lceil\frac{N}{L}\rceil$    |
| Work Group Size    | $L$                                 | $L+M-1$ |
| Total Work Items   | $N+2(M-1)$                          | $N(1+\lceil\frac{M-1}{L}\rceil)$    |
| Local Memory       | $L+M-1$ elements                    | $L+2(M-1)$ elements |
| Global Reads       | $N(1+\lfloor\frac{M-1}{L}\rfloor)$  | $N$ |
| Global Writes      | $N+M-1$                             | $N(1+\lfloor\frac{M-1}{L}\rfloor)$  |
| Work Efficiency    | $O(NM)$                             | $O(NM)$ |
| Step Efficiency    | $1$                                 | $K$ | 

<div><p></p></div>

## 4.3. Convolution in Parts
<div><p>
Both overlap-add and overlap-discard are generally good for convolving a long
sequence $I\in C^{N_1}$ with a short sequence $W\in C^{N_2}$ where $N_2\ll N_1$.
But what about convolving two long-sequences, as is the case in the backward
algorithm for $\frac{dE}{dW}$? Or what if the short sequence is actually kinda
long and, for example, $N_2$ is larger than the maximal value of $L$ (determined
by the hardware), or the size of $N_2$ is larger than what the implemented FFT
algorithm can efficiently handle?
</p></div>

<div><p>
In theory, the same guiding principles behind both the overlap methods can be
used to compute incrementally the convolution of two long sequences. The idea is
to divide one of the sequences into small chunks, apply one of the previous
methods on each of those chunks vs. the other long sequence - and accumulate the
results properly.
</p></div>

<div><p>
As usual, in practice things are not that simple. But let's start with the
theory though, and see how an idealized prototype of "Convolution in Parts"
algorithm looks like in Python. Assuming a method the computes the linear
convolution of one arbitrarily long sequence with another bounded sequence (of
length at most <code>MaxM</code>), we can write a method to compute the linear
convolution of two arbitrarily long sequences: 
</p></div>

**In [8]:**

{% highlight python %}
MaxL = 17

def linear_convolution(a, b):
    assert(len(b)<=MaxL)
    return np.convolve(a, b, mode='full') if len(b)>0 else np.zeros(len(a)-1)

def convolution_in_parts(a, b, L):
    convolution = np.zeros(len(a)+len(b)-1)
    for i in xrange(int(np.ceil(len(b)/L))+1):
        convolution[L*i:L*i+len(a)+L-1] += linear_convolution(a, b[i*L:(i+1)*L])
    return convolution
{% endhighlight %}

**In [9]:**

{% highlight python %}
a = np.random.normal(0.0, 1.0, 100)
b = np.random.normal(0.0, 1.0, 100)
assert(np.allclose(convolution_in_parts(a, b, MaxL), np.convolve(a, b, mode='full')))
{% endhighlight %}

 
<div><p>
Parallelizing this method seems problematic for the same issue we've encountered
with overlap-add: the final result is an accumulation of the sub-computations,
so they can't be parallelized without some synchronization. As a matter of fact,
this is often not a real problem here. If each call to <code>linear_convolution</code>
achieves full occupancy, then the structure of $\frac{|b|}{L}$ steps does not
really hurt the overall performances.
</p></div>

<div><p>
Still, when practical considerations enters the picture, and there are at-least
three reasons to refactor the above algorithm by  further decomposing the
computation, i.e. to explicitly partition both sequences, and compute all the
pairwise convolutions.
</p></div>

<div><p>
The first such reason comes from considering the use-case of computing the
minimal convolution, instead of the full linear convolution. This is the common
use-case in CNNs, where convolving two long sequences is required only for the
$\frac{dE}{dW}$ algorithm. Here, most of the computed values are discarded so
computing them is wasteful. The pairwise-convolutions approach leads to a much
more efficient algorithm.
</p></div>

<div><p>
A second reason comes from considering spectral algorithms, which are usually
preferable. In this case, the algorithm as structured above leads to
ridiculously excessive work if <code>linear_convolution</code> is implemented via a
spectral overlap method, since the Fourier transforms of the partitioning of the
long-sequence are computed anew in each step. It's much (much) more sensible to
perform a block-wise transformation of both sequences, then point-wise
multiplication between each pair of blocks, and finally inverting block-wise the
result.
</p></div>

<div><p>
Lastly, in some (real-life) scenarios such block-wise decomposition may lead to
better hardware utilization. For example, when the sequences are relatively
shorts, or when working in distributed architecture (e.g. grids and clusters).
</p></div>

<div><p>
The first step is to explicitly partition both sequences: 
</p></div>

**In [10]:**

{% highlight python %}
def blockwise_partitioning(a, b, L1, L2):
    blocks_a = [a[k*L1:(k+1)*L1] for k in xrange(len(a)/L1)]
    blocks_b = [b[k*L2:(k+1)*L2] for k in xrange(len(b)/L2)]
    return blocks_a, blocks_b
{% endhighlight %}
 
<div><p>
Then we go over all the $\frac{1}{2}(\frac{N_1}{L_1}\cdot\frac{N_2}{L_2})$ pair
of blocks, convolve them, and accumulate the results: 
</p></div>

**In [11]:**

{% highlight python %}
def spatial_blockwise_convolution(blocks_a, blocks_b, L1, L2):
    convolution = np.zeros(len(blocks_a)*L1 + len(blocks_b)*L2 - 1)
    for i, curr_a in enumerate(blocks_a):
        for j, curr_b in enumerate(blocks_b):
            convolution[i*L1+j*L2:(i+1)*L1+(j+1)*L2-1] += linear_convolution(curr_a, curr_b)
    return convolution
{% endhighlight %}

**In [12]:**

{% highlight python %}
a = np.random.normal(0.0, 1.0, 100)
b = np.random.normal(0.0, 1.0, 144)
L1, L2 = 10, 12
blocks_a, blocks_b = blockwise_partitioning(a, b, L1, L2)
assert(np.allclose(spatial_blockwise_convolution(blocks_a, blocks_b, L1, L2), np.convolve(a, b, mode='full')))
{% endhighlight %}

 
<div><p>
The algorithm executes $\frac{1}{2}(\frac{N_1}{L_1}\cdot\frac{N_2}{L_2})(L1+L2-1
)^2=\frac{1}{2}N_1N_2(\frac{L1}{L2}+2+\frac{L2}{L1}+\frac{1-2(L1+L2)}{L1L2})$
multiplications and additions. If $L1=L2$, that's $\sim
N_1N_2(2-\frac{4L-1}{2L^2})$ (this is not an approximation, but the tilda
notation). If $N_1$ and $N_2$ are at about the same length, that's essentially
equivalent to the direct spatial algorithm, which takes
$\sim(N_1+N_2-1)N_2\Rightarrow N_1N_2(1+\frac{N_2-1}{N_1})$ multiplications and
additions.
</p></div>

<div><p>
The spectral version of this method is a tricky business. Apparently, there are
two principle steps: (1) applying a block-wise Fourier transformation, (2)
pointwise multiplication of blocks and Fourier inversion. And indeed, simply
doing just that seems to work fine (recall from 1.4 that spectral convolutions
requires spatial paddings): 
</p></div>

**In [13]:**

{% highlight python %}
def blockwise_dft(blocks_a, blocks_b, L1, L2):
    pad = lambda block, K, M: np.pad(block, (0, (M-1)+(K-len(block))), mode='constant', constant_values=0.0)    
    dft_a = [np.fft.fft(pad(block, L1, L2)) for block in blocks_a]
    dft_b = [np.fft.fft(pad(block, L2, L1)) for block in blocks_b]
    return dft_a, dft_b

def spectral_blockwise_convolution(dft_a, dft_b, L1, L2):
    convolution = np.zeros(len(blocks_a)*L1 + len(blocks_b)*L2 - 1, np.complex128)
    for i, curr_a in enumerate(dft_a):
        offset = L1*i
        for j, curr_b in enumerate(dft_b):
            convolution[offset+j*L2:offset+(j+1)*L2+L1-1] += np.fft.ifft(curr_a*curr_b)
    return convolution
{% endhighlight %}

**In [14]:**

{% highlight python %}
a = np.random.normal(0.0, 1.0, 100)
b = np.random.normal(0.0, 1.0, 144)
L1, L2 = 10, 12
blocks_a, blocks_b = blockwise_partitioning(a, b, L1, L2)
dft_a, dft_b = blockwise_dft(blocks_a, blocks_b, L1, L2)
res = spectral_blockwise_convolution(dft_a, dft_b, L1, L2)
assert(np.allclose(res, np.convolve(a, b, mode='full')))
{% endhighlight %}

 
<div><p>
This is not very good. The code performs an inverse transform inside the inner
loop, and for each pair of blocks. That seems excessive. After all, the Fourier
transform is linear and all we're doing is addition - so it seems as if it
should be possible to apply the inverse transform in the same way the regular
transform is applied: once per block. So instead of
$\frac{1}{2}(\frac{N_1}{L_1}\cdot\frac{N_2}{L_2})$ inverse-transforms, we would
be doing just $\frac{N_1}{L_1}+\frac{N_2}{L_2}$ inverse-transforms. That could
lead to a significant speedup. For example, if $N_1=N_2=2^{20}$ and
$L_1=L_2=2^{10}$, that's about $256$ times faster.
</p></div>

<div><p>
That where the trickery is. In order for this idea to work, we must carefully
ensure that whenever we apply an inverse Fourier transform for an output
interval, then no input block that contributed to this interval had any
contributions outside of this interval. Otherwise, gibberish would result. The
interaction of the $i$-th block from the first sequence with the $j$-th block
from the second sequence contributes to the output elements whose positions are
in the range between $iL_1+jL_2$ and $(i+1)L_1+(j+1)L_2$. So in order to divide
the output elements into intervals such that each block-pair effects just one
interval, we must require $L_1|L_2$ (or $L_2|L_1$). This would give
$\frac{L_1}{L_2}+1$ different admissible divisions.
</p></div>

<div><p>
Before delving into the details, let's introduce a simple utility class whose
purpose is to keep some parameters and encapsulate some boilerplate code. The
most important thing to notice about it, are the <code>slice_start</code> and <code>slice_end</code>
parameters which are used to indicate the part of the linear convolution that is
actually required. For full convolution their values are $0$ and $N_1+N_2-1$
respectively, and the "valid" convolution their values are $N_1-1$ and $N_2$
respectively; but any other values (within the proper range) are acceptable.
The class maintains the assumptions and invariants according to which $N_1\le
N_2$ and $L_1|L_2$: 
</p></div>

**In [15]:**

{% highlight python %}
class ConvStructure(object):
    def __init__(self, N1, N2, L1, L2, slice_start, slice_end):
        self.N1, self.N2 = min(N1, N2), max(N1, N2)
        self.L1, self.L2 = min(L1, L2), max(L1, L2)
        self.slice_start, self.slice_end = slice_start, slice_end
        assert(self.L2 % self.L1 == 0)   
    
    # Returns the dimension of the resulting convolution
    def length(self):
        return self.slice_end-self.slice_start    
        
    # Returns the number of admissible divisions for the output.
    def divisions(self):
        return self.L2/self.L1+1

    # Returns the indices of the output interval effected by the
    # interaction between the i-th and the j-th blocks.
    def slices(self, i, j):
        output_begin = i*self.L1+j*self.L2
        output_end = output_begin + self.L1 + self.L2 - 1
        return output_begin, output_end      
    
    @staticmethod
    def full(N1, N2, L1, L2):
        N1, N2 = min(N1, N2), max(N1, N2)
        return ConvStructure(N1, N2, L1, L2, 0, N2+N1-1)

    @staticmethod
    def valid(N1, N2, L1, L2):
        N1, N2 = min(N1, N2), max(N1, N2)
        return ConvStructure(N1, N2, L1, L2, N1-1, N2)
{% endhighlight %}
 
<div><p>
Now we need a scheduling algorithm, to assign pairs of input-blocks to each
interval of each admissible division. This will determine the structure of
dependencies when parallelizing the algorithm: each relevant pair of input-block
would be assigned to a work-group, and any two work-groups could be executed
concurrently if and only if they belong to the same division but not the same
interval (the requirement regarding "same division" will be relaxed later).
</p></div>

<div><p>
The scheduling is based on the following observations: each divisions has
$\lceil\frac{N_1+N_2-L_2}{L_1+L_2}\rceil$ intervals. The $r$-th interval of the
$k$-th division begins at $kL_1 + r(L_1+L_2)$ and its length is $L_1+L_2-1$. It
is effected by the pairs $(i,j)$ that satisfy $iL_1+jL_2 = kL_1 + r(L_1+L_2)$.
Of course, any two blocks whose interaction does not effect the required
interval of the full linear convolution should not be computed.
</p></div>

<div><p>
Here's the scheduling logic in details: 
</p></div>

**In [16]:**

{% highlight python %}
# Returns:
#     workgroups[division_index][interval_index]  : List of pairs that contributes to this interval.
#     offsets[division_index]                     : The left-boundaries of the division's intervals.
def schedule_workgroups(structure):
    workgroups = []
    offsets = []
    first_intervals = []
    for division in xrange(structure.divisions()):
        first_intervals.append(max(int(np.floor((structure.slice_start+1-structure.L1*division)/(structure.L1+structure.L2+0.0))), 0))
        offsets.append(range(structure.L1*division+first_intervals[-1]*(structure.L1+structure.L2), structure.slice_end+1, structure.L1+structure.L2))
        workgroups .append([[] for interval in xrange(len(offsets[-1]))])

    for i in xrange(structure.N1/structure.L1):        
        lower = int(np.ceil(max(structure.slice_start-(structure.L1+structure.L2-1)-i*structure.L1, 0)/(structure.L2+0.0)))
        upper = int(np.floor((structure.slice_end-i*structure.L1)/(structure.L2+0.0)))
        upper = min(upper, structure.N2/structure.L2-1)
        if upper < 0:
            break               
        for j in xrange(lower, upper+1):
            first_index = i*structure.L1+j*structure.L2
            division = first_index%(structure.L1+structure.L2)/structure.L1
            interval = (first_index - division*structure.L1)/(structure.L1+structure.L2)
            workgroups[division][interval-first_intervals[division]].append((i, j))

    return workgroups, offsets
{% endhighlight %}

**In [17]:**

{% highlight python %}
workgroups, offsets = schedule_workgroups(ConvStructure.full(N1=60, N2=36, L1=12, L2=6))
for division_index, left_indices in enumerate(offsets):
    print 'Division %d Intervals:'%division_index
    for interval_index, left_boundary in enumerate(left_indices):
        print '\t %d\t : '%left_boundary, workgroups[division_index][interval_index]
{% endhighlight %}

**Out [17]:**
<pre>
    Division 0 Intervals:
    	 0	 :  [(0, 0)]
    	 18	 :  [(1, 1), (3, 0)]
    	 36	 :  [(0, 3), (2, 2), (4, 1)]
    	 54	 :  [(1, 4), (3, 3), (5, 2)]
    	 72	 :  [(4, 4)]
    	 90	 :  []
    Division 1 Intervals:
    	 6	 :  [(1, 0)]
    	 24	 :  [(0, 2), (2, 1), (4, 0)]
    	 42	 :  [(1, 3), (3, 2), (5, 1)]
    	 60	 :  [(2, 4), (4, 3)]
    	 78	 :  [(5, 4)]
    Division 2 Intervals:
    	 12	 :  [(0, 1), (2, 0)]
    	 30	 :  [(1, 2), (3, 1), (5, 0)]
    	 48	 :  [(0, 4), (2, 3), (4, 2)]
    	 66	 :  [(3, 4), (5, 3)]
    	 84	 :  []
</pre>
 
<div><p>
And the now for the algorithm itself. The following is a Python prototype that
is designed to "mimic the structure" of a real concurrent implementation (e.g.
in <code>OpenCL</code>). First, it has to perform a block-wise Fourier transform. This
is the job of the method <code>partitioning</code>.  Then, each admissible division
should be handled separately. It can be done concurrently, but with a price: we
will have to dedicate a separate buffer to each division - and at the end,
accumulate them. Those are two very common techniques for parallelism, namely
<b>map-reduce</b> (already mentioned in the FFTs section) and <b>privatization</b>.
</p></div>

<div><p>
This is actually a convenient situation: for small problems it wouldn't be a
problem to allocate the required space (which is of a factor of $\frac{L_2}{L_1}+1$
from the output size), and for large problems there would be enough intervals to
achieve full occupancy with a sequential processing of the admissible divisions.
For each division, the algorithm can now perform an accumulation of the point-
wise multiplications among the relevant input blocks, and only at the end
perform the inverse Fourier transforms: 
</p></div>

**In [18]:**

{% highlight python %}
class SpectralBlockwiseConvolution(object):
    @staticmethod
    def partitioning(a, b, structure):
        L1, L2 = structure.L1, structure.L2
        pad = lambda block, K, M: np.pad(block, (0, (M-1)+(K-len(block))), mode='constant', constant_values=0.0)
        blocks_a = [np.fft.fft(pad(a[k*L1:(k+1)*L1], L1, L2)) for k in xrange(len(a)/L1)]
        blocks_b = [np.fft.fft(pad(b[k*L2:(k+1)*L2], L2, L1)) for k in xrange(len(b)/L2)]
        return blocks_a, blocks_b

    @staticmethod
    def convolution(blocks_a, blocks_b, structure, workgroups, offsets):
        output_begin = max(structure.L1*int(np.floor(structure.slice_start / (structure.L1+0.0)))-(structure.L1+structure.L2), 0)
        output_end = min(structure.L1*int(np.ceil(structure.slice_end / (structure.L1+0.0)))+(structure.L1+structure.L2), structure.N1+structure.N2-1)
        convolution = np.zeros(output_end-output_begin, np.complex128)
        for division, division_workgroups in enumerate(workgroups):
            SpectralBlockwiseConvolution._execute_division(convolution, blocks_a, blocks_b, output_begin,
                                                           structure, division_workgroups, offsets[division])
        return convolution[structure.slice_start-output_begin:structure.slice_start-output_begin+structure.length()]

    @staticmethod
    def _execute_interval(output, blocks_a, blocks_b, output_offfset, structure, intervals_workers): 
        for workgroup in intervals_workers:
            (i, j) = workgroup
            output_begin, output_end = structure.slices(i, j)
            output[output_begin-output_offfset:output_end-output_offfset] += blocks_a[i]*blocks_b[j]
            
    @staticmethod
    def _execute_division(output, blocks_a, blocks_b, output_offfset, structure, division_workgroups, offsets):
        tmp = np.zeros(output.shape, np.complex128)
        for intervals_workers in division_workgroups:
            SpectralBlockwiseConvolution._execute_interval(tmp, blocks_a, blocks_b,
                                                           output_offfset, structure, intervals_workers)            
        for pos in offsets:
            position_begin = pos-output_offfset
            position_end = position_begin+structure.L1+structure.L2-1
            output[position_begin:position_end] += np.fft.ifft(tmp[position_begin:position_end])
{% endhighlight %}

**In [19]:**

{% highlight python %}
structure = ConvStructure(N1=6232, N2=12464, L1=19, L2=152, slice_start=511, slice_end=3283)

a = np.random.normal(0.0, 1.0, structure.N1)
b = np.random.normal(0.0, 1.0, structure.N2)

workgroups, offsets = schedule_workgroups(structure)
blocks_a, blocks_b = SpectralBlockwiseConvolution.partitioning(a, b, structure)
res = SpectralBlockwiseConvolution.convolution(blocks_a, blocks_b,
                                               structure, workgroups, offsets)

assert(np.allclose(res, np.convolve(a, b, mode='full')[structure.slice_start:structure.slice_end]))
{% endhighlight %}

 
<div><p>
So that's that. Does it worth it? Compared to the spatial version of
"convolution in parts", the answer is almost always "yes". But compared to the
"no hassle" approach of simply transforming the two sequences instead of doing
it block-wise the answer is sensitive to many many details such as the relative
size of sequences and the blocks, the absolute lengths of the sequences, the
optimization details of the FFT algorithm in use, the concurrency capabilities
of the hardware, etc.
</p></div>

<div><p>
Luckily, this does not concern us since in the only situation in which a CNNs
needs to convolve two long sequence (or images, etc) is in the $\frac{dE}{dW}$
algorithm - in which it doesn't require the full linear convolution, but only a
small of part if (whose length is the dimension of $W$). When this part is much
smaller than both $N_i$s (which is typically the case), then most of the blocks do
not interact; and when $|W|\lt L$, there are only $O(\frac{N}{L})$ interactions,
and the entire convolution takes $O(\frac{N}{L}L\log{L})=O(N\log{L})$. It
doesn't get more efficient than that: the direct spatial approach takes
$O(|W|N)$, and the direct spectral approach takes $N\log{N}$. 
</p></div>
 
<div><p>
Now the reason for preferring the <code>overlap-add</code> method over the <code>overlap-
discard</code> can be explained: it is reusability. Each training-phase of a
convolutional layer involves 3 convolutions: (1) The forward algorithm computes
$I\ast W$, (2) The $\frac{dE}{dI}$ backward algorithm computes
$\frac{dE}{dO}\ast W$, and (3) the $\frac{dE}{dW}$ backward algorithm computes
$\frac{dE}{dO}\ast I$. In order to carry the third computation by an overlapping
method, either $\frac{dE}{dO}$ or $I$ (and preferably, both) must be partitioned
into disjoint blocks (and each is a subject to Fourier transform). The only way
to compute the Fourier transform only once per block, is to use the overlap-add
method (which is based on a disjoint partitioning) on the first two
computations. The slight complications of overlap-add compared to overlap-
discard worths the computational benefits of reusable transformed sequences. 
</p></div>
