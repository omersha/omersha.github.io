---
layout: post
title: "Pointless Topology via Abstract Nonsense"
preview: "Trigger Warning: pure mathematics. Stone Duality gives a rigorous meaning to the slogan \"Geometry is dual to Algebra\"."
--- 
 
In 1940. G.H. Hardy published his essay "A Mathematician's Apology" in which he
used number theory as a prominent example for pure and inapplicable mathematics.
He meant it seriously, not as a joke. To the defense of his apology (pun a bit
intended), at the time only a handful of programmers thought about hash-tables,
cryptosystems were yet to be widely implemented in E-commerce platforms and not
many people used FFT before 1965 (no one but Gauss, I think). Indeed, computers
forced mathematicians to work harder and harder in order to keep annoying
engineers away from their offices.

This post is about Stone Duality: a theory that relates geometry and algebra
using a mathematical framework affectionately known as "abstract nonsense", and
gives rise to a perspective on topology which is self-admittedly pointless. This
is undoubtedly an heroic attempt to achieve pureness and inapplicability. Alas,
it turned out to be useful and practical as well. Many ideas about the design of
programming languages owe it their existence.

Yet, I'm not going to discuss applications. My motivation here is missionary:
everyone should know this stuff. It's not a matter of mathematical beauty
(though it is mathematically beautiful), but a matter of usefulness. Conceptual
usefulness, not practical. When I first studied it, I was really annoyed for not
studying it earlier, as it would have made so many things so much easier. In
terms of "Aha!-moments per second", the experience of studying it ranks high on
the list.

As abstract as this topic is (and I again refer the reader to the title), it's
still among the most concrete and explicit settings in which many concepts,
ideas and "mathematical slogans" can be demonstrated and motivated. The list is
long, and it contains (for example) spectrum of rings, algebraic probability,
topoi and "algebra is dual to geometry".

Before delving into the math, let's start with a soft overview using some heavy
amount of hand-waving. 
 
### 1. Overview
Generally, Stone duality relates geometric objects and algebraic objects.
Specifically, it establishes a precise relation between the category
$\mathrm{Top}$ of topological spaces with homeomorphisms between them and the
category $\mathrm{Loc}$ whose objects are certain type of lattices called
"Locales".

Taking a geometric perspective, this duality is motivated by thinking of a
"point" as an abstraction of the ability to zoom-in on an arbitrarily small
area. So points are not taken as fundamental entities, but rather as an
encapsulated limiting process: a "point" is identified with the collection of
all of its neighbourhoods.

Now, since the set of all well-defined locations of a space can be characterized
algebraically (more on this soon), we potentially may get 2 descriptions of the
same thing: as a space of points with some notion of proximity, and as an
algebraic structure of locations.

Alternatively, it's possible to start from an algebraic perspective by
considering the Lindenbaum–Tarski algebra of the propositional logic of
affirmative assertions. This requires some explaining.

The Lindenbaum–Tarski algebra of a logical theory $T$ is the quotient algebra
obtained by factoring the algebra of its formulas by the congruence relation
$p\sim q$ iff $p$ and $q$ are provably equivalent in $T$. For example, the
Lindenbaum–Tarski algebra of classical propositional logic, is Boolean algebra.

A logical assertion is affirmative if it's finitarily confirmable whenever it's
true (and refutative if it's finitarily disconfirmable whenever it's false). A
good model of such assertions is provided by thinking of physical rulers. Given
a ruler of length $N$, it may be possible to verify that an object is strictly
shorter then $N$, but due to measurement error, it's impossible to verify that
an object's length is exactly $N$. So the assertion "Zoidberg's height is less
then 160cm" is affirmative, while the assertion "Zoidberg's height is at least
160cm" is not affirmative (it might be true if his height is exactly 160cm, but
then it's not confirmable).

It's not hard to construct a logical theory for affirmative assertions, based on
the observational content of logical connectives. For example, a finite
conjunctions of affirmative assertions is an affirmative assertion, but an
infinite conjunctions of affirmative assertions is not affirmative (when it's
true, it requires an infinite number of tests to confirm). On the other hand, an
infinite disjunctions of affirmative assertions **is** an affirmative assertion
(whenever it's true, testing all those assertions sequentially until a true
assertion is found, is guaranteed to terminate after a finite number of steps).

Such reasoning can be applied to the other connectives to form a logical theory
called propositional geometric logic. Why "geometric"? This may be a good time
to recall the etymology of the term "geometry" (keeping in mind the rulers from
earlier): from Greek, combination of of "ge" ("earth, land") and "metria" ("a
measuring of"). "measurement of land".

This theory can be algebraically represented by the Lindenbaum–Tarski process
described above (this is a gross simplification that ignores technical details -
some of which I'll touch later). Again, this is a process of algebraizing
geometry (literally!), which is apparently equivalent to the previous one.

Both perspectives - the geometric and the logical - have their merits, and they
are priceless as a concretization of abstract thingies, whether within algebraic
geometry (e.g. Zariski topology, topoi) or within computer science (e.g. domain
theory). 
 
### 2. Categorical Preliminaries
Let's briefly explore some concepts, central for what's coming.

First, I would like to suggest that despite its name, Stone duality is not
necessarily about a duality. The way it relates geometry and algebra is
important and interesting in general, even when the conditions are too weak for
a proper duality. This weaker (yet rich) relation is **adjunction**. This is a
categorical notion that relates two functors via a pair of natural
transformations, which are themselves a way to relate two functors.

I won't explain categories and functors here: this is common knowledge, and the
concepts are easy to understand simply by reading the definitions (which can be
found anywhere). But I will say something about natural transformations and
adjunctions, which might be somewhat less familiar and are certainly harder to
grasp.

A natural transformation is a relation between two functors that share domain
and codomain $F_1, F_2: C\rightarrow D$ (here $C$ and $D$ are categories). A
natural transformation $T:F_1\rightarrow F_2$ assigns each object $X$ in $C$, a
morphism $\eta_X$ in $D$, such that the so-called "naturality condition" hold:
for any $(f:X\rightarrow Y)\in C$ the following diagram must commute:

$$\require{AMScd}
\begin{CD}
F_1(X) @>F_1(f)>> F_1(Y)\\
@V\eta_XVV @VV\eta_YV\\
F_2(X) @>F_2(f)>> F_2(Y)
\end{CD}$$

It's very instructive to think of a [natural transformation as a
homotopy](http://mathoverflow.net/questions/64365/natural-transformations-as-
categorical-homotopies) between the functors. So if $I$ denotes the interval
category $0\rightarrow 1$, then a natural transformation $T:F_1\rightarrow F_2$
is a functor $I\times C\rightarrow D$ such that $T(0,X)=F_1(X)$ and
$T(1,X)=F_2(X)$. Another useful mental model, is to think of categories as
graphs. The graph indexing category $\mathrm{GrIn}$ is the category $\mathrm{Ar}
\underset{\text{tgt}}{\overset{\text{src}.}{\rightrightarrows}} \mathrm{Ve}$. In
this light, a functor $G:\mathrm{GrIn}\rightarrow\mathrm{Set}$ is a graph, and a
natural transformation of two such functors is precisely the same thing as a
graph homomorphism.

As for adjunctions, wikipedia
[says](https://en.wikipedia.org/wiki/Adjoint_functors) "adjunction is ubiquitous
in mathematics". It also quotes Mac Lane: "adjoint functors arise everywhere".
Be it as it may, I must admit I found it exceptionally tricky to grok when I
first encountered it. But well, I ain't no mathematician.

So what's the big idea? The formal definitions are dry and unenlightening. The
ingredients are two functors $\mathrm{A}
\underset{G}{\overset{F}{\rightleftarrows}} \mathrm{B}$, two natural
transformations, $\eta:1_A\rightarrow G\circ F$ (the *unit*) and
$\epsilon:F\circ G\rightarrow 1_B$ (the *counit*), and conditions to relate
them, namely "the triangle identities" $(\epsilon F)\circ(F\eta)=1_F$ and
$(G\epsilon)\circ(\eta G)=1_G$. This definition blurs the asymmetry between $F$
and $G$, but it's fundamental: $F$ is left-adjoint to $G$ and $G$ is right-
adjoint to $F$ (this is denoted $F\dashv G$).

The remedy for an unhelpful definition is a good example. Traditionally,
adjunctions are introduced by an example of a free functor (as the left-adjoint)
and a forgetful functor (as the right-adjoint). Such pairs are an absolutely
common and important mathematical theme, but I find their pedagogical value
doubtful. To their credit, they do make it easy to remember that generally left-
adjoints "increase" and right-adjoints "decrease".

Instead, here are two examples that made it click for me.

The [first example](http://mathoverflow.net/questions/6551/what-is-an-intuitive-
view-of-adjoints-version-1-category-theory) deals with numbers. Let $Z$ and $Q$
denote the integers and the rationals as categories whose morphisms are given by
$\le$, and consider the inclusion $I:Z\rightarrow Q$ as a functor. Then floor
$x\mapsto\lfloor x\rfloor$ is a functor $Q\rightarrow Z$ which is right-adjoint
to $I$, and ceil $x\mapsto\lceil x\rceil$ is a functor $Q\rightarrow Z$ which is
left-adjoint to $I$. I'll prove the left-adjointness for ceil, and leave floor
as an exercise (that is to say, I'm lazy):

We have to provide 2 natural transformations, counit $\epsilon:C\circ
I\rightarrow 1_Z$ and unit $\eta:1_Q\rightarrow I\circ C$ such that
$1_C=\epsilon C\circ C\eta$ and $1_I=I\epsilon\circ\eta I$. The counit
$\epsilon$ is trivial: $C\circ I:Z\rightarrow Z$ is $1_Z$ (it takes an integer,
"cast it" as a rational number and ceil it, and it's trivially monotone).

As for the unit $\eta$, the functor $I\circ C:Q\rightarrow Q$ ceils a rational
number, and "casts it" back to $Q$, and it maps $x\le y$ to $\lceil
x\rceil\le\lceil y\rceil$ in $Q$. This is a natural transformation, since for
all $x\le y$ the following naturality diagram commutes:

$$\require{AMScd}
\begin{CD}
1_Q(x) @>I_Q(x\le y)=(x\le y)>> 1_Q(y)\\
@V\eta(x)=(x \le\lceil x\rceil) VV @VV\eta(y)=(y \le\lceil y\rceil)V\\
I\circ C(x)=\lceil x\rceil @>I\circ C(x\le y)=(\lceil x\rceil\le \lceil
y\rceil)>> I\circ C(x)=\lceil y\rceil
\end{CD}$$

Essentially, the natural morphism of $\eta$ is $x\Rightarrow x\le\lceil
x\rceil$, and it's easy to check that the triangle identities hold
($1_C=\epsilon C\circ C\eta\Leftrightarrow (x\le x)(C(x\le\lceil
x\rceil))=x\le\lceil x\rceil$ etc).

The second example also deals with order. This is not a coincidence. As a matter
of fact, order theory is by many means a simplified version of category theory
(a perspective that has its own name,
"[decategorification](http://ncatlab.org/nlab/show/decategorification)"). And
since order theory is central to Stone duality, let's take this opportunity to
recall some definitions.

A **preorder** is a reflexive and transitive relation, a **proset** is a set
equipped with a preorder, a **partial order** is an antisymmetric preorder, and
a **poset** is a set equipped with a partial order. The process that takes
prosets to posets (via the equivalence $x\equiv y$ iff $x\le y$ and $y\le x$) is
functorial, and left-adjoint to the forgetful-functor
$\mathrm{Poset}\rightarrow\mathrm{Preorder}$. If $X$ is a set with some order,
then the set with the opposite order is denoted $X^\text{op}$.

The **meet** of two elements $x,y$, denoted $x\land y$, is their greatest lower
bound, or infimum (it might not exist), and the **join** of two elements,
denoted $x\lor y$, is their least upper bound, or supremum (it might also not
exist). Obviously, join and meet are dual (meaning, $z=x\land y$ in $X$ if and
only if $z=z\lor y$ in $X^\text{op}$). In fact, they are the decategorification
of the categorical notions of limit and colimit.

A poset in which any two elements have both a unique meet and a unique join is
called a **lattice**. When all arbitrary sets are bounded, the lattice is said
to be **complete**. An easy fact with deep consequences, is that lattices are
algebraic structures. So if we're given a poset with infimums and supremum, we
may replace the partial-order with 2 binary operations, and lose no information
($x=\land\{x,y\}\Leftrightarrow x\le y\Leftrightarrow y=\lor\{x,y\}$). The
deepness is due to the analogy between lattices and Cartesian-closed categories,
and it has much to do with Stone duality.

We're ready for the example: Let $X$ be a lattice, and let $\delta:X\rightarrow
X\times X$ be the diagonal mapping $x\mapsto(x,x)$. Then it has a (unique)
right-adjoint, the meet operation $\land:X\times X\rightarrow X$ and a (unique)
left-adjoint, the join operation $\lor:X\times X\rightarrow X$. Essentially, the
meet operation is a right-adjoint because $(a,a)\le(x,y)\Leftrightarrow a\le
x\land y$, and the join operation is a loft-adjoint because $x\lor y\le
z\Leftrightarrow(x,y)\le(z,z)$.

To conclude, I'll just mention that the composition of two adjoint functors give
rise to a monad, and that any monad is representable as an adjunction. It should
be noted that while the discussion above focused on adjoint functors, adjunction
is a concept that applies generally in 2-categories. The idea is the same (a
prototypical 2-category is the category $\mathrm{Cat}$ of small categories as
objects, functors as morphisms and natural-transformation as 2-cells).

This should feel familiar to programmers. After all, a monad is just a monoid in
the category of endofunctors. No problem. 
 
### 3. Heyting Algebras
The central objects of Stone duality are Heyting algebras. Those are the
"intermediate objects" used to relate topologies and locales. Such structures
come up naturally in 2 different contexts, and induce a nontrivial connection
between them:
1. As the structure of the Lindenbaum–Tarski algebra of some logical theories.
2. As the structure of the open-sets of all topological spaces.

Explicitly, an Heyting algebra is a distributive lattice with implications. A
lattice is distributive whenever meets distribute over joins $x\land(y\lor
z)=(x\land y)\lor(x\land z)$ (like unions distribute over intersections), and
implications are defined via the equational relation $z\le x\rightarrow y
\Leftrightarrow z\land x\le y$. As a programmer at heart, I intuitively think of
Heyting algebras as "exponentiated lattices" and of implications as analogous to
Currying: $\mathrm{hom}(Z,Y^X)\cong\mathrm{hom}(Z\times X, Y)$.

In the logical context, the underlying order reflects entailments and
implications are, well, implications. The algebraization of propositional
intuitionistic logic is a Heyting algebra, and the algebraization of
propositional geometric logic is a complete Heyting algebra (shortly, cHa).
While the first result is easy, the second is not. The difference is due to
finitary presentations. This is a rather technical subject, so I will only
review it briefly.

An algebraic theory $T$ consists of some operations (of arbitrary arities) and
some laws (which are equations $e_1=e_2$ where the $e_i$s are compositions of
the operations). A $T$-algebra is an instantiation of $T$ in terms of a carrier
set and functions. There are many instantiations of a theory (e.g. there are
many groups in group theory), so how could one single out a specific
instantiation of interest?

The algebraic approach is to use presentations. A presentation $T\langle G|R
\rangle$ is a set of generators $G$ and a set of equational relations $R$. There
could be many $T$-algebras that model a given presentation, so we still haven't
singled out a specific instantiation. But we could try looking for the most
"restrictive" model in the sense that it factors through any other model, and
call it **the** model of the presentation. When such a universal construction
exists, we say that this $T$-algebra is presented by the presentation $T\langle
G|R \rangle$.

Often, this is well and sound. For example, if $T$ is a finitary algebraic
theory, then any presentation $T\langle G|R \rangle$ presents a $T$-algebra -
and this is the case for propositional intuitionistic logic. But propositional
geometric logic is not finitary. Fortunately, it still presents (and its
presentations are complete Heyting algebras), but proving this takes some
effort.

In a sharp change of direction, a topology $\Omega X$ is also a complete Heyting
algebra. This is not obvious: It is surely a lattice (w.r.t. unions and
intersections as joins and meets). And by definition, arbitrary joins of open
sets are open. But arbitrary meets of open sets are not necessarily open - so
how come is it **complete**? And where are the implications?

Time for a methodological detour. There are 3 different categories whose objects
are complete Heyting algebras:
1. $\mathrm{cHa}$: In which morphisms preserve both arbitrary joins and
arbitrary meets.
2. $\mathrm{Frm}$ (Frames): In which morphisms preserve arbitrary joins, but
only finite meets.
3. $\mathrm{Loc}$ (Locales): In which morphisms preserve arbitrary meets, but
only finite joins.

This is certainly non-obvious, and implies: **(A)** As a structure, being either
a Frame or a Locale is equivalent to being a cHa. **(B)** A morphism may
preserve arbitrary joins, but not arbitrary meets (and vice versa). I won't
prove it here, but just hint that given a frame, we can define implications by
$a\rightarrow b := \lor\{c:c\land a\le b\}$, and that actually stronger
statements hold (e.g. a lattice is a frame if and only if it is cHa). Also, note
that the categories $\mathrm{Frm}$ and $\mathrm{Loc}$ are dual.

Back to topology. We showed above that open-sets have the structure of a frame,
and now we know that any frame is a cHa. So topologies are indeed structured as
complete Heyting algebras. But this is hardly satisfactory. How, concretely,
should we understand infinitary meets of open sets?

Let $V_i$ be an infinite sequence of open sets. Then in general $\land V_i$
won't be given as an intersection of open sets. What then? It should satisfy
$\land V_i\le V_j$, and if $W\le V_i$ for all $i$ then $W\le\land V_i$. So
$\land V_i$ should be the "largest" open set containing $\cap V_i$. That is,
take the arbitrary union of all the open sets that are disjoint to $\cap V_i$,
and obtain a open set. Its complement is a closed set whose interior is $\land
V_i$. Implications then, are given by $U\rightarrow V := \mathrm{int}(U^c\cup
V)$. 
 
### 4. Localification and Spatialization
Alright, now we're in business. We are about to construct a **localification**
functor $\mathrm{Lc}:\mathrm{Top}\rightarrow\mathrm{Loc}$ that abstracts away
the points of topological spaces and a **specialization** functor
$\mathrm{Sp}:\mathrm{Loc}\rightarrow\mathrm{Top}$ that maps locales to the
"simplest" topological space they describe. Then we'll show those two functors
are adjoint: $\mathrm{Lc}\dashv\mathrm{Sp}$.

From the discussion above it is clear that we have a **contravariant** functor
from $\mathrm{Top}$ to $\mathrm{Frm}$: it sends a continuous function
$f:X\rightarrow Y$ (which is a morphism in $\mathrm{Top}$) to a morphism
$f^{-1}:\Omega Y\rightarrow\Omega X$ in $\mathrm{Frm}$. This is just the
definition of continuity.

Equivalently, we have a **covariate** functor
$\mathrm{Lc}:\mathrm{Top}\rightarrow\mathrm{Loc}$. This calls for some
clarification: this functor maps $f:X\rightarrow Y$ to the opposite arrow of
$f^{-1}:\Omega Y\rightarrow\Omega X$, which is a morphism $\Omega
X\rightarrow\Omega Y$ in $\mathrm{Loc}$, but this is a formal construction that
seems to say nothing about this arrow as a map. Yet, it does: this arrow should
preserve arbitrary meets and finite joins, and generally those arrows should
respect (opposite) compositions w.r.t to this contravariant functor.

Localification of a topological space is an abstraction, and it leads to
information loss: many different topological spaces may induce the same locale
(easy example: any set equipped with the trivial topology $\{X,\emptyset\}$
induces the same locale. So it's impossible in general to restore the original
points of the space from a locale).

However, given a locale we can still obtain a spatial object (i.e. a space with
points) which is categorically optimal. This requires a recipe to (optimally)
represent points of a locale $L$ (which is a "pointless space") together with a
topological structure that is consistent with the locale. The topological space
$\mathrm{Sp}(L)$ is called "the **spectrum** of $L$". The topology of the
spectrum is a locale $\mathrm{Lc}(\mathrm{Sp}(L))$ which is homomorphic to $L$
(i.e. there is a locale-morphism $L\rightarrow\mathrm{Lc}(\mathrm{Sp}(L))$).

Actually, instead of a recipes - how about 4? The following are 4 suitable
approaches - each has its own ups and downs. All of those constructions "work"
in a categorical sense: locale-morphisms preserve points.


#### Recipe 1: Points as Frame Morphisms
Given a space $(X,\Omega X)$, there is a natural one-one correspondence between
points $x_0\in X$ and functions $f_{x_0}:\{0\}\rightarrow X$ given by $0\mapsto
x_0$. The set $\{0\}$ has a unique topology (the trivial one,
$\{\{0\},\emptyset\}$), thus we have a canonical $\mathrm{Frm}$-morphism
$f_{x_0}^{-1}:\Omega X\rightarrow\Omega\{0\}$.

The construction is easily interpretable: Since the elements of $L$ are meant to
model "locations", then a point can be naturally thought of in terms of all the
locations containing it. So a point can be identified with a characteristic
function $\mu:L\mapsto\{0,1\}$ that preserve arbitrary meets and finite joins
(finite, since an arbitrary join of non-neighborhoods may be a neighborhood),
i.e., $\mu$ is a $\mathrm{Loc}$-morphism. It's dual is exactly the
$\mathrm{Frm}$-morphism constructed above.

To specify $\mathrm{Sp}:\mathrm{Loc}\rightarrow\mathrm{Top}$, we need to
functorially map $L$ to open sets defined over those points. The object part is
constructed by associating each $\ell\in L$ with an open set
$\phi(\ell):=\{\mu\in\mathrm{Sp}(L) | \mu(\ell)=1\}$ (it's not hard to show that
this is indeed a topology - and it should be intuitively clear by the above
interpretation regarding points).

As for the morphism-part, we need to describe how to functorially associate a
$\mathrm{Loc}$-morphism $\mu:L\rightarrow M$ with a continuous function
$f\_\mu:\mathrm{Sp}(L)\rightarrow\mathrm{Sp}(M)$. Since we model points in
$\mathrm{Sp}(L)$ and $\mathrm{Sp}(M)$ as $\mathrm{Frm}$-morphisms, $f\_\mu$ will
map points by composing them with the $\mathrm{Frm}$-morphism
$\mu^\mathrm{op}:M\rightarrow L$, so $f\_\mu(x):=x\circ\mu^\mathrm{op}$. This
definition "compiles": any $x\in\mathrm{Sp}(L)$ is a $\mathrm{Frm}$-morphism
$L\rightarrow\{0,1\}$, so $x\circ\mu^\mathrm{op}$ is a $\mathrm{Frm}$-morphism
$M\rightarrow\{0,1\}$, which means
$f\_\mu(x)=x\circ\mu^\mathrm{op}\in\mathrm{Sp}(M)$.


#### Recipe 2: Points as Completely Prime Filters
The second construction is also easily interpretable. First, recall the
definitions:

* A Filter $\mathscr{F}$ on a lattice $X$ is a non-empty subset of $X$ such that
$\mathscr{F}=\uparrow G$ for some $G\subseteq X$ and
$x,y\in\mathscr{F}\Rightarrow x\land y\in\mathscr{F}$.
* Filter $\mathscr{F}$ on a lattice $L$ is **Prime** iff $a_1\lor
a_2\in\mathscr{F}$ implies that $a_1\in\mathscr{F}$ or $a_2\in\mathscr{F}$.
* Filter $\mathscr{F}$ on a lattice $L$ is **Completely Prime** iff the above
holds for arbitrary joins: $\bigvee{a_i\in\mathscr{F}}\Rightarrow\exists a_i$
s.t. $a_i\in\mathscr{F}$.

Thinking again of a "point" as a collection of locations, filters become an
obvious candidates to model points. The restriction to completely-prime filters
meant to mitigate the issue (mentioned above) regarding arbitrary-joins and can
be understood as allowing only neighbourhood systems as admissible filters
(indeed, in topological spaces any collection $\mathscr{U}\_x:=\{U\in\Omega X |
x\in U\}$ is a completely-prime filter).

This construction is equivalent to the first one: Any completely-prime filter
$\mathscr{F}\subset L$ can be associated with the locale-morphism
$\mu:L\rightarrow\{0,1\}$ s.t. $\mu(x)=1\Leftrightarrow x\in F$, and any locale-
morphism $\mu:L\rightarrow\{0,1\}$ induced the completely-prime filter
$\mathscr{F}:=\{x\in L | \mu(x)=1\}$.

The object part of $\mathrm{Sp}:\mathrm{Loc}\rightarrow\mathrm{Top}$ is
constructed by associating each $\ell\in L$ with an open set
$\Sigma\_\ell:=\{\mathscr{F}\subset L | \ell\in\mathscr{F}\}$ where $\mathscr{F}$
is completely prime. This is a topology, since $\Sigma_0=\emptyset$,
$\Sigma_1=\{\mathscr{F} | \mathscr{F}\subset L\}$ (completely prime filters),
$\Sigma\_{a\land b}=\Sigma_a\cap\Sigma_b$ and
$\Sigma\_{\bigvee{a_i}}=\bigcup\Sigma\_{a_i}$.

The morphism-part is constructed as following: let $\psi:L\rightarrow M$ be a
locale-morphism, and define
$\mathrm{Sp}(\psi):\mathrm{Sp}(L)\rightarrow\mathrm{Sp}(M)$ by
$\mathscr{F}\mapsto(\psi^\*)^{-1}(\mathscr{F})$ where $\psi^\*$ is the left Galois
adjoint of $\psi$ (and since $\psi^\*$ is a $\mathrm{Frm}$-morphism and
$\mathscr{F}$ is completely prime over $L$, then $(\psi^\*)^{-1}(\mathscr{F})$ is
completely prime over $M$ - so $\mathrm{Sp}(\psi)$ maps points to points). The
map $\mathrm{Sp}(\psi)$ is continuous, since
$(\mathrm{Sp}(\psi))^{-1}(\Sigma\_a)=\Sigma\_{\psi^\*(a)}$, thus $(\mathrm{Sp}(\psi))^{-1}(\Sigma\_a)=\{\mathscr{F} | \mathrm{Sp}(\psi)(\mathscr{F})=(\psi^\*)^{-1}(\mathscr{F})\in\Sigma\_a\}=\{\mathscr{F}|a\in(\psi^\*)^{-1}(\mathscr{F})\}=\{\mathscr{F}|\psi^\*(a)\in\mathscr{F}\}=\Sigma\_{\psi^\*(a)}$.


#### Recipe 3: Points as Principal Prime Ideals
The third construction models points as principal prime ideals. It is equivalent
to the second one, since principal prime ideals are dual to completely prime
filters (heuristically, a principal prime ideal is a minimal collection of
"negligible sets" that contains a point).


#### Recipe 4: Points as Meet-Irreducible Element
The forth construction essentially follows from the third: an element $\ell\in
L$ is meet-irreducible whenever the principle ideal $\downarrow\ell$ is also
prime. Alternatively, it follows from the second construction: if $\mathscr{F}$
is a completely-prime filter, then $\ell\_\mathscr{F}:=\bigvee\{x\in L | x
\not\in \mathscr{F}\}$ is meet-irreducible, and if $\ell\in L$ is meet-
irreducible, then $\mathscr{F}\_\ell:=\{x\in L | x\not\le\ell\}$ is a completely-
prime filter. Moreover, $\ell\_{\mathscr{F}\_\ell}=\ell$ and
$x\in\mathscr{F}\_{\ell\_\mathscr{F}}\Leftrightarrow x\in\mathscr{F}$.

Now the functor $\mathrm{Sp}:\mathrm{Loc}\rightarrow\mathrm{Top}$ is very simple
(but less "interpretable"). The object part is given by $\ell\in L\mapsto\{p\in
L | \ell\not\le p\}$ (so $L$ is mapped into a topology), and the morphism part
is $(\psi:L\rightarrow M)\mapsto(\{p\in L | \ell\not\le p\}\mapsto\{\psi(p) |
p\in L, \ell\not\le p\})$ which is continuous.

#### The Adjunction
First, as promised, let's see that locale-morphisms preserve points. It seems
easiest to show that any locale-morphism $\mu:L\rightarrow M$ sends meet-
irreducible elements to meet-irreducible elements: So let $\ell\in L$ be meet-
irreducible, and assume that for $x,y\in M$ we have $x\land y\le\mu(\ell)$.

Since $L$ is a complete lattice and $\mu$ preserves arbitrary meets, it is the
upper-component in a unique Galois connection $(\mu^\*, \mu)$ (that is,
$\mu^\*:M\rightarrow L$ and $\mu^\*(a)\le b\Leftrightarrow a\le\mu(b)$). Thus
$x\land y\le\mu(\ell)$ implies $\mu^\*(x)\land\mu^\*(y)=\mu^\*(x\land y)\le\ell$,
and since $\ell$ is meet-irreducible in $L$ then assuming w.l.g
$\mu^\*(x)\le\ell$ implied $x\le\mu(\ell)$ and $\mu(\ell)$ is meet-irreducible in
$M$.

Finally - the functors $\mathrm{Lc}:\mathrm{Top}\rightarrow\mathrm{Loc}$ and
$\mathrm{Sp}:\mathrm{Loc}\rightarrow\mathrm{Top}$ are adjoint, $\mathrm{Lc}$ to
the left and $\mathrm{Sp}$ to the right, with unit
$\lambda:\mathrm{Id}\_\mathrm{Top}\rightarrow\mathrm{Sp}\circ\mathrm{Lc}$ and
counit $\sigma:\mathrm{Lc}\circ\mathrm{Sp}\rightarrow\mathrm{Id}\_\mathrm{Loc}$.

The natural transformation corresponds to the counit $\sigma$ is:

$$\require{AMScd}
\begin{CD}
(\mathrm{Lc}\circ\mathrm{Sp})(L) @>f_{|\{\mathscr{F}\}}>>
(\mathrm{Lc}\circ\mathrm{Sp})(M)\\
@V\sigma_L:=(\phi_L)_*VV @VV\sigma_M:=(\phi_M)_*V\\
L @>f>> M
\end{CD}$$

Where $L,M$ are locales, $f:L\rightarrow M$ is a locale-morphism,
$f\_{|\{\mathscr{F}\}}$ is $f$ considered as a set-function restricted to
completely prime filters over $L$, and
$\phi_L:L\rightarrow(\mathrm{Lc}\circ\mathrm{Sp})(L)$ is a
$\mathrm{Frm}$-morphism defined as $\ell\mapsto\Sigma\_\ell$ (so
$\sigma_L:=(\phi_L)\_\*$ is its right Galois adjoint, hence a
$\mathrm{Loc}$-morphism).

Recall that $(f^\*)^{-1}:\mathrm{Sp}(L)\rightarrow\mathrm{Sp}(M)$, where $f^\*$ is
the left Galois adjoint to $f$, maps continuously completely prime filters over
$L$ to completely prime filters over $M$, so $((f^\*)^{-1})^{-1}=f^\*$ maps open-
sets from $\Omega\mathrm{Sp}(M)$ to open sets from $\Omega\mathrm{Sp}(L)$, which
makes a $\mathrm{Frm}$-morphism, thus $f\_{|\{\mathscr{F}\}}$ is a
$\mathrm{Loc}$-morphism.

The natural transformation corresponds to the unit $\lambda$ can be constructed
similarly. 
 
### 5. Spatiality and Sobriety
So far, we have constructed an adjunction - which is truly great. But we were
promised a duality!

A duality is expected when the topology is rich enough to render points useless.
So obviously, the axiom $T_0$ is a precondition for the possibility of
reconstructing points from locations. But it's not enough: different $T_0$
spaces may still have the same locales.

One way to proceed, is by looking for a stronger separation axiom (but as weak
as possible), that allows restoring the points of a space from its topology.
Turns out, this separation axiom is $T_D$: for each $x\in X$ there is an open
$x\in U$ such that $U/\{x\}$ is open as well (note: $T_1\Rightarrow
T_D\Rightarrow T_0$).

Indeed, for $T_D$ spaces, we have the following theorem (Thron): Let $X$ be a
$T_D$-space, let $Y$ be a $T_0$-space and let $\Phi:\Omega(Y)\cong\Omega(X)$ be
a lattice isomorphism. Then there is precisely one continuous mapping
$f:X\rightarrow Y$ such that $\Phi = \Omega(f)$. Consequently, if both $X$ and
$Y$ are $T_D$ then the $f$ is a homeomorphism.

But apparently, $T_D$ spaces are not very useful (such are the rumors, anyway).
A better characterization is that of **sobriety**: a space $X$ is said to be
sober if there are no meet-irreducible open sets other then $X$ ("No
neighbourhood systems without a point"). It's also true that a space $X$ is
sober if and only if the filters $U(x)$ are precisely the completely prime ones.

The constructions above should have made clear that "sobriety works": let $Y$ be
sober and let $X$ be arbitrary. Then the continuous maps $f:X\rightarrow Y$ are
in a one-one correspondence with the $\mathrm{Frm}$-morphisms
$h:\Omega(Y)\rightarrow\Omega(X)$ (given by $f\mapsto f^{-1}$).

Sobriety is not a separation axioms, but rather, it is a type of "completeness":
zooming-in always leads to a point. There are no "missing point" in Sober
spaces. $T_D$ does not imply Sobriety, and Sobriety does not imply $T_D$. But
any Hausdorff space is sober, any finite $T_0$-space is sober, any Noetherian
Alexandroff Space is sober, and (my sources tell me) posets with the Scott
topologies are typically sober. So sobriety is common.

A locale $L$ is spatial only when it is isomorphic to the topology of its own
spectrum. Otherwise, it is not isomorphic to the topology of any space. So $L$
is spatial iff $\sigma_L:\mathrm{Lc}(\mathrm{Sp}(L))\rightarrow L$ is a complete
lattice isomorphism. $\mathrm{Sp}(L)$ is always sober.

Historically, Stone duality was first proved for the special case of locales
which are Boolean algebras. Stone was motivated by his work on functional
analysis: he was interested in a spectral representation theory for operators on
Hilbert spaces, and noted that abstractly, Boolean algebras are equivalent to
Boolean rings, which are "spaces of projections" (all their elements are
idempotent).

Stone showed that any Boolean algebra is spatial, and its spectrum is a
topological space which is rather exotic: it is compact, totally disconnected
and Hausdorff (such spaces are now called "Stone spaces"). The importance of
this theorem resides in the fact that it leads to representation theorem for
Boolean algebras: any abstract Boolean algebra is isomorphic to a concrete
Boolean algebra (i.e. a field of sets). 
