---
layout: post
title: "Meta-Sequences in C++"
preview: With the introduction of variadic templates to C++, meta-sequences became a central idiom in meta-programming. The standard implementation is not always the best choice.
--- 

The relation between C++ and C is somewhat of an anomaly; usually abstraction
hurts performance, yet C++ code often compiles to a more efficient executable
than its C equivalent. Templates deserve a lot of the credit for this, as they
allow the compiler to resolve at compile-time much of the abstraction required
by modular and maintainable code design. That's what meta-programming is all
about.

In that respect, [variadic
templates](http://en.cppreference.com/w/cpp/language/parameter_pack), introduced
in C++11, were a major improvement of the language. And with them came a new and
extremely useful idiom: meta-sequences. Below I'll give a demonstration of a
problem - initializing an array at compile-time - for which meta-sequences
provide a good solution.

Since they are the key idea behind many elegant solutions to various problems,
it is not surprising that utilities for using them were included in the
standard library (since [C++14](http://en.cppreference.com/w/cpp/utility)). It
is surprising, though, that the implementation that comes with the STL is sometimes
problematic and may require using homemade alternatives (this is the case, for
example, in gcc 5.1.0). So I'll discuss implementation details as well. 
 
### Compile-Time Array Initialization 
 
As a simple scenario, consider the problem of initializing a
```std::array<T,N>``` when ```T``` is a non-trivially constructible type:

{% highlight cpp %}
struct DummyType
{
    DummyType(int arg) : val(arg) {}
    int val;
};
auto pool = std::array<DummyType, 3>(); // Error: no matching function for call to ‘DummyType::DummyType()’
{% endhighlight %}

All kinds of runtime hacks (e.g. switching to an array of pointers and
constructing its elements in a loop) are obviously evil. For example, they
introduce an extra level of indirection that obstructs the optimizer, and they
present an unattractive choice between heap allocations or custom
allocators.

Luckily, this problem has an easy and decent solution:

{% highlight cpp %}
auto pool = MakeArray<3>([](size_t index){return DummyType(index);});
{% endhighlight %}

where the function ```MakeArray``` will be defined shortly. This code is
evaluated, in compile time, to this:

{% highlight cpp %}
auto pool = std::array<int,N> {DummyType(0), DummyType(1), DummyType(2)};
{% endhighlight %}

So the compiler is in a position to eliminate copies, moves and function calls,
and generate in-place consturctions wherever possible. As a matter of fact, with
this main function -

{% highlight cpp %}
int main()
{
   auto pool = MakeArray<12>([](size_t index){return DummyType(index*2);});
   return pool[3].val;
}
{% endhighlight %}

the entire computation was done by the compiler (using gcc 5.1.0), and the
generated assembly code was essentially:

{% highlight asm %}
mov     eax, 6
ret
{% endhighlight %}
 
### The Function "MakeArray" 
 
So how could a function like ```MakeArray``` be implemented? Using meta-
sequences of course. In C++14 ```std::integer_sequence``` became a standard part
of the language, as part of the header ```<utility>```. Two very useful helpers
are the templates ```std::index_sequence``` and ```std::make_index_sequence```.

Here's an implementation for ```MakeArray``` that uses them:

{% highlight cpp %}
template<typename Ctor, size_t... S>
std::array<std::result_of_t<Ctor(size_t)>, sizeof...(S)> MakeArray(Ctor&& ctor, std::index_sequence<S...>)
{
   return std::array<std::result_of_t<Ctor(size_t)>, sizeof...(S)> {ctor(S)...};
}

template<size_t N, typename Ctor>
std::array<std::result_of_t<Ctor(size_t)>, N> MakeArray(Ctor&& ctor)
{
   return MakeArray(std::forward<Ctor>(ctor), std::make_index_sequence<N>());
}
{% endhighlight %}

Let's look at the first template. The first argument is a callback that
returns elements to be placed in the array (here, it takes an
index as a parameter), and the second argument has no name. This means the
function never uses its value. Only its type matters. The method returns an
array whose elements' type is the same type returned by the callable given as
the first argument, and whose length is equal to the length of the variadic
integer sequence ```S```.

The value of ```S``` is deduced from the second, nameless, argument. In fact,
that's the raison d'etre of that second argument. It has no other purpose. The
template ```index_sequence<>``` is an empty class, which is nothing more than a
placeholder for the sequence ```S```.  In essence, this is it:

{% highlight cpp %}
template<size_t...> struct index_sequence {};
{% endhighlight %}

A call for the first template of ```MakeArray``` looks like that:

{% highlight cpp %}
auto arr = MakeArray(ctor, std::index_sequence<0,1,2,3,4,5,6,7,8,9,10,11>());
{% endhighlight %}

The function ```make_index_sequence``` is a mechanism for constructing a
sequence such as ```index_sequence<0,1,2,3,4,5,6,7,8,9,10,11...,N-1>``` from the
integer ```N```. It is used by the second template, whose job is to allow the
user to do the same by simply writing:

{% highlight cpp %}
auto arr = MakeArray<12>(ctor);
{% endhighlight %}

And this is it.

That was just one scenario in which meta-sequences shine. There are many others.
But I think it's best not to make this post into a long list of random examples,
and it's likely that later posts will provide real-life scenarios anyway. So
instead, let's go on to discuss some implementation details. 
 
### Meta-Enumeration - Take 1 
 
Here's a naive implementation of ```make_index_sequence```:

{% highlight cpp %}
template<size_t N, size_t... S> struct make_sequence_imp : make_sequence_imp<N-1, N-1, S...> {};
template<size_t... S> struct make_sequence_imp<0, S...> {using type = index_sequence<S...>;};
template<size_t length> using make_index_sequence = typename make_sequence_imp<length>::type;
{% endhighlight %}

The third line is merely syntactic sugar. The first line defines a template
class that encodes a postfix of the final sequence (```S...```) and a parameter
```N``` that counts the missing elements. This class is defined recursively, so that
shorter postfixs are derived from longer postfixs. The second line is a
specialized version of this class for the base case, of a fully spanned sequence
- and it defines the type ```type``` that encodes the entire sequence, and is
accessible (due to inheritance) from all the chain of derived classes.

For example, unrolling $N=4$ get us:

{% highlight cpp %}
struct make_sequence_imp<0, 0, 1, 2, 3> {using type = index_sequence<0, 1, 2, 3>;};
struct make_sequence_imp<1, 1, 2, 3> : make_sequence_imp<0, 0, 1, 2, 3> {};
struct make_sequence_imp<2, 2, 3> : make_sequence_imp<1, 1, 2, 3> {};
struct make_sequence_imp<3, 3> : make_sequence_imp<2, 2, 3> {};
struct make_sequence_imp<4> : make_sequence_imp<3, 3> {};
{% endhighlight %}

So the type of ```make_sequence_imp<4>::type``` is ```index_sequence<0, 1, 2, 3>```.

Now the problem with this implementation might be clearer: generation of a
sequence of $N$ elements requires an inheritance tree of depth $N$. Furthermore,
sequences of different lengths can't share intermediate instantiations. This
could easily make compile time unreasonably long even for moderate $N$s.

Surprisingly, even though there is a better solution at hand - this kind of
solution is used by [common implementations of the STL](https://github.com/gcc-
mirror/gcc/blob/master/libstdc%2B%2B-v3/include/std/utility)! 
 
### Meta-Enumeration - Take 2

A faster and a compiler-friendlier approach, is to use concatenation:

{% highlight cpp %}
template<typename Seq1, typename Seq2> struct Concatenate;

template<size_t... S1, size_t... S2>
struct Concatenate<index_sequence<S1...>, index_sequence<S2...>> {
    using type = index_sequence<S1..., S2...>;
};
{% endhighlight %}

Using it has several benefits: it doesn't require a recursive inheritance, it
compiles in $O(\ln n)$ time instead of $O(n)$ and it uses reusable building-
blocks for the sequences shareable between different instantiations. Here's an
implementation for concatenation-based sequences via ranges:


{% highlight cpp %}
template<size_t first, size_t length>
struct make_range_imp;

template<size_t first, size_t length>
using make_range = typename make_range_imp<first, length>::type;

template<size_t first, size_t length>
struct make_range_imp {
   using type = typename Concatenate<make_range<first, length/2>,
make_range<first+length/2, length-length/2>>::type;
};

template<size_t first>
struct make_range_imp<first, 0> {
   using type = index_sequence<>;
};

template<size_t first>
struct make_range_imp<first, 1> {
   using type = index_sequence<first>;
};
{% endhighlight %}

And we can now write:

{% highlight cpp %}
template<size_t length> using make_index_sequence = make_range<0, length>;

auto tmp = make_index_sequence<1000>(); // Ok!
auto tmp = std::make_index_sequence<1000>(); // Compilation error!
{% endhighlight %}
