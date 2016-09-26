---
layout: post
title: "Random Integers"
preview: Efficient sampling from a discrete distribution is a useful yet nontrivial algorithmic building-block, which involves some interesting and clever ideas.
---


<div><p>
The need to sample from a discrete distribution given by a probability vector
$\vec{p}=(p_1,p_2,...,p_n)$ comes up very often. For example, when taking a
bootstrapped sample from a weighted dataset, or when selecting parents from a
population based on their realtive fitness in genetic algorithms, or when
implementing an agent that applies a mixed strategy, or... well, I made my
point: often.
</p></div>

<div><p>
There are of course standard libraries that provide such functionality. For
example, C++ has <code>std::discrete_distribution</code>, and python has
<code>numpy.random.choice</code>. But that was not always the case. In C++
this function was added ~2011 and numpy has it since ~2013. It's still quite
common to work in a setting in which such functions are unavailable (C++98,
JavaScript, x86...) - so knowing the gory details of how to draw random
numbers is still useful.
</p></div>

<div><p>
Back in the day, when I interviewed programmers, this was one of my favorite
interview discussions. I used to ask the candidate to read Vose's paper (see
below), implement the algorithm it presents, and explain how it works. Not a bad
filter. 
</p></div>
 
#### Outline
1. [Sources of Randomnes](#Randomness)
2. [Han-Hoshi Algorithm](#HanHoshi)
3. [Radix-Based Lookup Tables](#LookupTables)
4. [The Alias Method](#AliasMethod)

## <a name="Randomness"></a> 1. Sources of Randomness

<div><p>
Any random algorithm must have access to some source of randomness. Algorithms
for drawing from a categorical distribution usually assume either the ability to
flip a fair coin, or an access to a random source of uniform numbers from
$[0,1)$.
</p></div>

<div><p>
Neither of those assumptions is restrictive, in the sense that an access to an
(stationary) arbitrary source of randomness can be used to simulate both of them.
To see this, first note that (ignoring precision issues) given
$X\sim\mathrm{Uniform}[0,1)$ we can obtain $Y\sim\mathrm{Bernoulli}(\frac{1}{2})$
by $Y\le 0.5\Rightarrow X=0$ and $Y\gt 0.5\Rightarrow X=1$, and that given
$Y\sim\mathrm{Bernoulli}(\frac{1}{2})$ we can obtain $X\sim\mathrm{Uniform}[0,1)$
(assuming, for example, a Q1.b fixed-point system) by sampling $b$ bits for
the fractional part of $X$ from $Y$.
</p></div>

<div><p>
So those assumptions are theoretically interchangeable. In practice, though, the
type a available randomness may effect the computational efficiency of an
algorithm.
</p></div>

<div><p>
Now, an access to an arbitrary (though stationary) source of randomness provides
an access to random bits, which can be thought of as a (possibly biased) coin flips.
How can we use it to simulate a fair coin? And how can we be sure the a given
coin is indeed fair?
</p></div>

<div><p>
<a href="https://en.wikipedia.org/wiki/John_von_Neumann">Von-Neumann</a> ("the
Simpsons of science" - whatever you're thinking, he already did it) gave a
simple answer: given any coin, with an unknown fixed bias $q$, it's possible to
simulate a fair coin as following: flip the coin twice. On $\mathrm{HT}$ output
"1", on $\mathrm{TH}$ output "0", and otherwise repeat.

The algorithm halts after $\frac{1}{q(1-q)}$ steps on average:
</p></div>

{% highlight python %}
def unfair_demonstration(coin_flip, max_iters):
    for i in xrange(max_iters):
        C = coin_flip()
        if C != coin_flip():
            return C
    return 0
{% endhighlight %}

<div><p>
In the case where $q$ is known beforehand, it's possible to construct a faster
algorithm - but never mind that. The point is that it's not hard to build a
piece of hardware or a simple software that approximates a flip of a fair coin
in constant time.
</p></div>

<div><p>
As for the reversed scenario, of simulating biased coin using a fair one - I
discussed excatly this in the post <a
href="http://www.trivialorwrong.com/2015/12/01/random-bitstreams.html">Random
Bitstreams.</a> 
</p></div>
 
## <a name="HanHoshi"></a> 2. Han-Hoshi Algorithm
<div><p>
The central question in this post is: how can one simulate the categorical
distribution given by the stochastic vector $\vec{p}=(p_1,p_2,...,p_n)$?
</p></div>

<div><p>
A good case-study for such algorithms is the <strong>Han-
Hoshi</strong> algorithm. Let's start with assuming an access to a fair coin:

<ol>
 <li>Let $Q_i=\sum_{k=1}^np_k$ (and $Q_0=0$), and start with the interval
$I_0=[0,1]$.</li>
 <li>In each step $t$ flip a coin, and choose the next interval $I_{t}$ to be the
left-half of $I_{t-1}$ or the right-half of $I_{t-1}$ based on the result.</li>
<li> Stop when $I\subset[Q_{i-1},Q_i)$ for some $i$. Output $i$.</li>
</ol>

Or in code:
</p></div>

{% highlight python %}
def han_hoshi_bernoulli(probabilities):
    assert(np.isclose(np.sum(probabilities), 1))

    K = len(probabilities)
    i, j, I, Q = 0, K, (0, 1), [0, 1]

    while True:
        if i > j:
            return i-1
        random_bit = np.random.choice([0,1])
        if random_bit == 0:
            I = (I[1]-(I[1]-I[0])/2.0, I[1])
            while I[0] >= Q[0]:
                Q[0] += probabilities[i]
                i += 1
        elif random_bit == 1:
            I = (I[0], I[0]+(I[1]-I[0])/2.0)
            while I[1] <= Q[1]:
                j -= 1
                Q[1] -= probabilities[j]
{% endhighlight %}

<div><p>
It's not hard to understand why it's correct, but don't worry if you don't
immediately see it. It will be clarified soon.
</p></div>

<div><p>
Under the assumption of an access to a fair coin, this algorithm is near-optimal
in the following sense: $H_2(\vec{p})\le E[T]\le H_2(\vec{p})+3$ where $T$ is
the number of steps and $H_2(\vec{p}):=-\sum{p_i\log_2(p_i})$ is the binary
entropy of $\vec{p}$.
</p></div>

<div><p>
This is asymptotically optimal, since the definition of <a
href="https://en.wikipedia.org/wiki/Entropy_(information_theory)">entropy</a>
was designed to model the expected amount of "coin flips" that produce the same
randomness as a given random variable. So if we (and by "we" I mean <a
href="https://en.wikipedia.org/wiki/Claude_Shannon">Shannon</a>) have defined
"entropy" correctly, then $H_2(\vec{p})$ should give the number of iterations
required to simulate a categorical distribution. The content of the
<a href="https://en.wikipedia.org/wiki/Shannon%27s_source_coding_theorem">source
coding theorem</a> is pretty much a statement that the definition indeed works.
</p></div>

<div><p>
But if we'd obtain access to a uniform source of randomness, we could improve
this result. Not only that, but the algorithm also becomes simpler:

<ol>
<li> Let $\vec{Q}=(0, p_1,p_1+p_2,...,\sum{p_i})$ be the CDF of $\vec{p}$.</li>
<li> Produce a random number $x\in[0,1)$.</li>
<li> Return $i$ such that $P_i\le x$ and $P_{i+1}\gt x$.</li>
</ol>

Or in code:
</p></div>

{% highlight python %}
def han_hoshi_uniform(probabilities):
    assert(np.isclose(np.sum(probabilities), 1))

    K = len(probabilities)
    Q = np.hstack(([0], np.cumsum(probabilities)))
    return bisect.bisect_right(Q, np.random.uniform(0, 1))-1
{% endhighlight %}

<div><p>
This is the first algorithm you'd probably come up by yourself if you had to -
and I hope it clarifies why the Bernoulli version was correct. Its running-time
is independent of $H_2(\vec{p})$: it has an initialization step that takes
$\Theta(n)$ (to generate $\vec{Q}$), and each draw amounts to a binary search in
the ordered array $\vec{p}$, which takes $\Theta(\ln n)$.
</p></div>

<div><p>
The fact that by assuming an access to a uniform random-number generator we were
able to remove the dependency of the running-time from the entropy of the
distribution may seem a bit weird, and rightly so. This is actually an artifact
of some implicit (and natural) assumptions taken by the second algorithm,
regarding the number-system in use (finite-precision rationals, v.s. real
numbers) and the finiteness of $\vec{p}$.
</p></div>

<div><p>
Note that this it's not necessarily a speedup, e.g. by considering 
$\vec{p}\in[0,1]^{1000}$ with $p_0=1$. Usually, though, the second algorithm is
faster. And its speedup is (roughly) caused by the fact that the random number
$x\in[0,1)$ provides $b$ random-bits at once, while the bounds for the entropy
of $\vec{p}$ are controlled by the same $b$.
</p></div>

<div><p>
The assumption at hand from here onward will be the availability of a uniformly
random source $X\sim\mathrm{Uniform}[0,1)$. We'll see we can do much better than Han-Hoshi.
</p></div>
 
## <a name="LookupTables"></a> 3. Radix-Based Lookup Tables
<div><p>
A good place to start the exploration is by paying with space to improve time.
The naive way of doing so would be to allocate an array whose cells are assigned
with labels according to the relative frequencies specified by $\vec{p}$, and
then uniformly draw a cell and return its label. This would allow sampling in
$O(1)$, but the induced costs of the initialization time and space-requirements
are too high to make this idea useful.
</p></div>

<div><p>
For example, the distribution $\vec{p}=(0.125, 0.375, 0.05, 0.45)$ induces the
frequencies $\vec{c}=(\frac{5}{40}, \frac{15}{40}, \frac{2}{40},
\frac{18}{40})$, so drawing from $\vec{p}$ can be achieved by returning the
content of the $\lfloor 40X \rfloor$-th cell from the following array:

<table class="array"><tr>
<td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1</td><td>1</td><td>1</td>
<td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td>
<td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td>
<td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>2</td>
<td>2</td><td>3</td><td>3</td><td>3</td><td>3</td><td>3</td><td>3</td><td>3</td>
<td>3</td><td>3</td><td>3</td><td>3</td><td>3</td><td>3</td><td>3</td><td>3</td>
<td>3</td><td>3</td><td>3</td><td>3</td><td>3</td><td>3</td><td>3</td><td>3</td>
<td>3</td><td>3</td><td>3</td><td>3</td><td>3</td>
</tr></table>
</p></div>

<div><p>
The trouble caused by the space-requirements is obvious, but it's worth noting
that the efficiency of the initialization step is most definitely not
negligible, since in practice the probabilities $\vec{p}$ are not fixed, and the
initialization step must be re-executed often.
</p></div>

<div><p>
This idea, however, can be turned into a useful algorithm nonetheless, and it
can also provide some further insights about the interaction between the
numerical precision used for the probabilities, the entropy $H_2(\vec{p})$ and
number of categories $n$. The key is to observe that the length of the array
described above depends on all of those 3 factors: The length must obviously be
larger than $n$, a larger entropy $H_2(\vec{p})$ pushes for larger common
denominator in the frequencies-representation of $\vec{p}$, hence for a longer
array, and the precision in which the $p_i$s are represented bounds this common
denominator.
</p></div>

<div><p>
We'll work with fixed precision: choose a radix $\beta$, and let $k$ be the
numbers of digits used to represent each of the $p_i$s. And since the discussion
below becomes much simpler when framed around absolute-counting than around
relative-frequencies will do just that. So, continuing with the example above,
if $k=3$ and the base is decimal ($\beta=10$), we'll consider the distribution
induces by the counting data $(125, 375, 50, 450)$ instead of the distribution
$(0.125, 0.375, 0.05, 0.45)$. We lose nothing by doing that.
</p></div>

<div><p>
Now, the counts can be decomposed according to the significance of that digits -
$$(125, 375, 50, 450) = (100, 300, 0, 400) + (20, 70, 50, 50) + (5, 5, 0, 0)$$
and each can be associated with a "relative-frequency array" as above.
</p></div>

<div><p>
So $(100, 300, 0, 400)$ (that represents $\frac{800}{1000}$ of the population)
will be associated with -


<table class="array"><tr><td>0</td><td>1</td><td>1</td><td>1</td><td>3</td><td>3</td><td>3</td
><td>3</td></tr></table>
</p></div>

<div><p>
And $(20, 70, 50, 50)$ (that represents $\frac{190}{1000}$ of the population)
will be associated with -

<table class="array"><tr><td>0</td><td>0</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td
><td>1</td><td>1</td><td>2</td><td>2</td><td>2</td><td>2</td><td>2</td><td>2</td
><td>3</td><td>3</td><td>3</td><td>3</td><td>3</td></tr></table>
</p></div>

<div><p>
And $(5, 5, 0, 0)$ (that represents $\frac{10}{1000}$ of the population) will be
associated with  -

<table class="array"><tr><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1</td><td>1</td
><td>1</td><td>1</td><td>1</td></tr></table>
</p></div>

<div><p>
It's an easy observation that we will always have $k$ such vectors, and that the
lengths of all the "relative-frequency arrays" are $\le(\beta-1)n$. This
naturally suggests the following algorithm:

<ol>
<li> Construct $k$ such relative-frequencies arrays.</li>
<li> Compute the "weights" for each of those arrays.</li>
<li> On sampling: pick randomly one of those arrays based the distribution
computed at step 2, and pick a label from it uniformly.</li>
</ol>
</p></div>

<div><p>
The code steps 1 and 2 (that is, the initialization) looks like that -
</p></div>

{% highlight python %}
def make_tables(probabilities, k):
    def make_table(pos, digits):
        table = [i for i, d in enumerate(digits) for j in xrange(d)]
        p = np.sum(digits)/(10.0**(pos+1))
        return table, p

    n = len(probabilities)
    digit_weights = np.zeros(k+1)
    tables = []
    for j in xrange(k):
        table, digit_weights[j+1] = make_table(j, [int(np.floor(probabilities[i]*(10**(j+1)))%10) for i in xrange(n)])
        tables.append(table)
    digits_cdf = np.cumsum(digit_weights)
    return tables, digits_cdf
{% endhighlight %}

<div><p>
Step 3 makes it a looks like a recursive algorithm (the problem of picking an
array is the original problem), but actually, we can perform it using the
"uniform version" of Han-Hoshi from earlier, and since $k$ is small and constant
(i.e. independent of $n$), this step is $O(1)$. So this entire algorithm takes
$\theta(n)$ for initialization, $O(1)$ for sampling and its space complexity is
$O((\beta-1)nk)=O(n)$.
</p></div>

{% highlight python %}
def lookup_draw(tables, digits_cdf):
    table = tables[bisect.bisect_right(digits_cdf, np.random.uniform(0, 1))-1]
    return table[np.floor(np.random.uniform(0, 1)*len(table)).astype(np.int32)]
{% endhighlight %}

<div><p>
This algorithm can be very efficient in common real-life scenarios, and be used
to simulate many general distributions (binomial, geometric...) by approximate
them using a stochastic vector (which is often better than using their pdf and
allows to explicitly control the time-space trade-off). 
</p></div>
 

## <a name="AliasMethod"></a> 4. The Alias Method
<div><p>
The alias method is a really great algorithm. Probably one of my all-time
favorites. It is clever, neat, simple and useful.
</p></div>

<div><p>
Let's again consider the example $\vec{p}=(0.125, 0.375, 0.05, 0.45)$. If we
were to draw from the uniform discrete probability $(0.25, 0.25, 0.25, 0.25)$
instead of from $\vec{p}$ - how wrong will it make us? Well, for each category
the mistake would have been $\frac{p_i}{0.25}$. So one thing we can try doing to
fix this error, is applying some-kind of a component-wise <a href="https://en.wikipedia.org/wiki/Rejection_sampling">rejection
sampling</a>.
</p></div>

<div><p>
This is not that simple though. If we choose a category uniformly at
random, we would like to return it with probability $q_i$ such that $\frac{1}{n}q_i=p_i$, which implies
$q_i=np_i$. But the algorithm "(1) choose $i$ uniformly at random. (2) with
probability $\min(np_i,1)$ return $i$, or else report failure" will work
correctly only for the categories $i$ such that $np_i\le 1$; but will grossly
undercount the other categories.
</p></div>

<div><p>
We can hope to fix it by switching from "report failure", to returning the
under-counted categories in a way that would compensate their under-sampling. In
our running example, the starting point is -

<ol>
<li> Choose a category $i\in\{0,1,2,3\}$ uniformly at random.</li>
<li> If $i=0$, return $0$ with probability
$\min{(\frac{0.125}{0.25},1)}=\frac{1}{2}$, or else, in probability
$1-\frac{1}{2}=\frac{1}{2}$... [do what?].</li>
<li> If $i=1$, return $1$ with probability $\min{(\frac{0.375}{0.25},1)}=1$ -
it's under-sampled ($0.25$ instead of $0.375$).</li>
<li> If $i=2$, return $2$ with probability
$\min{(\frac{0.05}{0.25},1)}=\frac{1}{5}$, or else, in probability
$1-\frac{1}{5}=\frac{4}{5}$... [do what?].</li>
<li> If $i=3$, return $3$ with probability $\min{(\frac{0.45}{0.25},1)}=1$ -
it's under-sampled ($0.25$ instead of $0.45$).</li>
</ol>
</p></div>

<div><p>
Which seems exceptionally lucky: we have to find an event that occurs with
probability $0.375-0.25=0.125$ to return $1$, in order to achieve to appropriate
probability for this category, and we just happen to have one available in the
case $i=0$ was chosen, but not returned (this indeed happens in probability
$\frac{1}{4}(1-\frac{1}{2})=0.125$). Similarly, we have to find an event that
occurs with probability $0.45-0.25=0.2$ to return $3$, in order to achieve to
appropriate probability for this category, and again we just happen to have one
available in the case $i=3$ was chosen, but not returned (this indeed happens in
probability $\frac{1}{4}(1-\frac{1}{5})=0.2$).
</p></div>

<div><p>
So the following algorithm works perfectly in this example:

<ol>
<li> Choose a category $i\in\{0,1,2,3\}$ uniformly at random.</li>
<li> If $i=0$, return $0$ with probability $\min{(\frac{0.125}{0.25},1)}=\frac{1}{2}$, or else in probability
$1-\frac{1}{2}=\frac{1}{2}$ return $1$.</li>
<li> If $i=1$, return $1$ with probability $\min{(\frac{0.375}{0.25},1)}=1$</li>
<li> If $i=2$, return $2$ with probability $\min{(\frac{0.05}{0.25},1)}=\frac{1}{5}$, or else in probability
$1-\frac{1}{5}=\frac{4}{5}$ return $3$.</li>
<li> If $i=3$, return $3$ with probability $\min{(\frac{0.45}{0.25},1)}=1$.</li>
</ol>
</p></div>

<div><p>
What a lucky coincidence!
</p></div>

<div><p>
Well, this is not a coincidence, <a href="http://unsongbook.com/">because nothing ever
is</a>. We can always arrange things so it would work. This
was shown by Walker, popularized by Knuth who offered an $O(n\ln n)$ algorithm
that does this initialization (see TACOP, Volume 2, 3.4.1), and was improved by 
<a href="https://web.archive.org/web/20131029203736/http://web.eecs.utk.edu/~vose/Publications/random.pdf">Vose</a>
to an $O(n)$ algorithm. So we get an initialization
step that takes $O(n)$, and afterwards we can draw in $O(1)$. The space
requirements are $2n$.
</p></div>

<div><p>
Both the sampling algorithm itself and its initialization step (together known
as the "Alias Method") works pretty much like in the example above. The sampling
algorithm is:

<ol>
<li> Maintain an array of length $n$, denoted $P$, called "Probabilities". Entries
are in $[0,1]$.</li>
<li> Maintain an array of length $n$, denoted $A$, called "Alias". Entries are in
$\{0,1,...,n-1\}$.</li>
<li> Obtain a random number $x\in[0,n)$. If $P[\lfloor x\rfloor]\lt x-\lfloor
x\rfloor$ return $\lfloor x\rfloor$. Otherwise return $A[\lfloor x\rfloor]$.</li>
</ol>
</p></div>

<div><p>
The initialization step, which consists of constructing $P$ and $A$, is:

<ol>
<li> Divide $\vec{p}$ into "small" and "large" probabilities (realtive to
$\frac{1}{n}$).</li>
<li> The small probabilities can be assigned a $P_i$ value: $np_i$ where $p_i$ is
their original probability.</li>
<li> For each element $k$ with a large probability:
   <ol>
    <li> Pick a small element $j$ with an uninitialized alias, and make $k$ its
alias.</li>
    <li> Update $p_k\leftarrow p_k-\frac{1}{n}(1-np_j)$</li>
    <li> If $p_k$ is now small, assign $P_k=np_k$ (and flag it as a "small element with no alias").</li>
    <li> Else repeat the above.</li>
    </ol>
</li>
</ol>
</p></div>

<div><p>
The idea of this algorithm can be visualized using "square histograms": A
square histogram is what you get by taking the histogram of $\vec{p}$ and "put
on it" the flipped histogram of $1-\vec{p}$. So it's a square of height 1,
divided into equal columns, and each column is divided into 2 blocks. In the
current context, the bottom histogram is of "Probabilities" and the upper
histogram is for the conditional probabilities of the "Aliases".
</p></div>

{% highlight python %}
# Highly Unoptimized. Hopfully readable.

def init_vose(probabilities):
    probabilities = probabilities.copy()
    n = len(probabilities)
    uniform = 1/(n+0.0)

    smalls = [i for i,p in enumerate(probabilities) if p < uniform]
    larges = [i for i,p in enumerate(probabilities) if p >= uniform]

    probs = np.zeros(n)
    alias = np.zeros(n, dtype=np.int32)
    while len(smalls) > 0:
        small = smalls.pop()
        probs[small] = n*probabilities[small]
        if len(larges) > 0:
            large = larges.pop()
            alias[small] = large
            probabilities[large] -= uniform*(1-probs[small])
            if probabilities[large] > uniform:
                larges.append(large)
            elif probabilities[large] > 0:
                smalls.append(large)
    return probs, alias

def draw_vose(vose_probabilities, vose_aliases):
    n = len(vose_probabilities)
    x = np.random.uniform(0,n)
    i = int(np.floor(x))
    return i if x-i < vose_probabilities[i] else vose_aliases[i]
{% endhighlight %}

<div><p>
There are several optimizations that a realistic implementation needs to
consider. For starters, multiplications of the form $pn$ - which are an integral
part of the algorithm - are often faster when $n$ is a power of $2$ (which
reduces the multiplication to bit-shifting). This can speedup the general-case,
since employing a zero-padding to $\vec{p}$ does not change the distribution.
But note that the newly introduced probability-zero categories may still have
aliases assigned to them.
</p></div>

<div><p>
More important is the efficient usage of the random numbers with respect to the
available numerical precision. For example, the implementation above can call
the number random generator once, since it assumes it returns a floating point
number in the range $(0,n]$.
</p></div>

<div><p>
Generally though, the random numbers $x$ will be
drawn from $(0,1]$ and their MSBs $x_0$ will be used for the index (as
$i:=\lfloor nx_0\rfloor$) while their LSBs $x_1$ will be used to select the
category from the 2 available options (using the condition $P[i]\lt x_2$). Such
strategy requires a suitable random-number generator (e.g. so that the LSBs won't
be biased), and a large enough precision compared to $n$ and the precision in
which the probabilities $\vec{p}$ are given (this is rarely a problem, but still
requires acknowledgment).
</p></div>

<div><p>
Moreover, a slight modification of the algorithm, by appropriately choosing the
factors it uses, leads an algorithm which is a bit more numerically stable and
can handle unnormalized probability-vectors, which is very helpful in practice.
</p></div>

<div><p>
The following code demonstrates the idea, but a production-suitable implementation
should take care of how to update the unnormalized distributions and maintain their
normalization factors:
</p></div>


{% highlight python %}
def init_modified_vose(probabilities):
    probabilities = probabilities.copy()
    normalization_factor = np.sum(probabilities)
    n = len(probabilities)
    baseline = normalization_factor/(n+0.0)
    
    smalls = [i for i,p in enumerate(probabilities) if p < baseline]
    larges = [i for i,p in enumerate(probabilities) if p >= baseline]
    
    probs = np.ones(n)*baseline
    alias = np.zeros(n, dtype=np.int32)
    while len(smalls) > 0:
        small = smalls.pop()        
        probs[small] = probabilities[small]
        if len(larges) > 0:
            large = larges.pop()
            alias[small] = large
            probabilities[large] -= baseline*(1-n*probs[small])
            if probabilities[large] > baseline:
                larges.append(large)
            elif probabilities[large] > 0:
                smalls.append(large)
    return probs, alias, normalization_factor


def draw_modified_vose(vose_probabilities, vose_aliases, normalization_factor):
    n = len(vose_probabilities)
    x = np.random.uniform(0,n)
    i = int(np.floor(x))
    return i if (x-i)*normalization_factor < n*vose_probabilities[i] else vose_aliases[i]
{% endhighlight %}
