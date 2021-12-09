---
title: Notes on Information theory and Coding
date: 2020-10
tags:
    - Information Theory
    - Cambridge
    - Note
---
# Entropy

Entropy is used to charactize the information of a **distribution**. Let $X$ be a random variable, $P(x) = Pr(X=x)$. Entropy is defined as:

$$H(X) = \sum_xP(x)\log\frac{1}{P(x)}$$

## Information and Expected Information



## Properties of Entropy

There are three important entropy for entropy:

1. Entropy is non-negative
2. If $x$ can take on $M$ values, there must be $H(x) \leq \log(M)$ 
3. The inequality above only holds when $P(x)$ is a equiprobable distribution with probability $\frac{1}{M}$ for each $x \in X$

Non-negativity can be proved by IT inequality. The latter two can be proved by using Langrange multiplier or the IT inequality.

> IT inequality: $\ln x \leq x-1$, equality holds at $x=1$

## Joint Distribution and Conditional Entropy

The entropy of a joint distribution is defined as:

$$H(x,y) = \sum_{x,y}P_{XY}(x,y)\log\frac{1}{P_{XY}(x,y)}$$

straight from the definition, the conditional entropy of $Y$ conditioned on $X$ is defined as:

$$H(y|x) = \sum_{x,y}P_{XY}(x,y)\log\frac{1}{P_{Y|X}(y|x)}$$

> In the second equality, the preceding multiplier is still the joint probability because the expected information sent out is still based on a random pick in the joint probability space, only that we are using the conditional probability to calculate the actual information.

In computation, we often use $H(Y|X) = \sum_{x\in X}p(x)H(Y|X=x)$

The below equality is true:

$$H(x,y) = H(x)+H(y|x) = H(y)+H(x|y)$$

and can be proved by the product rule $P(x,y)=P(y|x)P(x)$



# Estimating Tail Probability

Given a random variable $X$, we are often interested in estimating the tails of its distribution (i.e., $P(x>C)$) given some information about the distribution (mean, variance, etc.) There are two inequalities that allows us to do that. 

## Markov's Inequality

If $X$ is a **non-negative** random variable with mean $\mu$, then we have

$P(X>a) \leq \frac{\mu}{a} $

The proof is quite simple. 

$$ E(x) =  \int_{-\infty}^{a} xp(x)dx + \int_{a}^{\infty}xp(x)dx $$
$$ E(x) \geq  \int_{-\infty}^{a} xp(x)dx + \int_{a}^{\infty}ap(x)dx $$
$$ E(x) \geq \int_{-\infty}^{a} xp(x)dx + ap(x\geq a)$$
$$ E(x) \geq ap(x>=a) $$
$$ p(x\geq a) \leq \frac{E(x)}{a} $$

## Chebyshev's Inequality

If $X$ is a random variable with mean $\mu$ and variance $\sigma^2$, then for $a>0$

$$ P(|X-E(x)|>=a) = \frac{\sigma^2}{a^2} $$

This follows straight from Markov's inequality by setting $Y = (X-EX)^2$ and notice that $\sigma^2 = E[(X-EX)^2] = EY$

# Weak Law of Large Numbers (WLLN)


Roughly speaking, WLLN is "Emperical average converges to the mean", rigorously:

Let $X_1, X_2, ...$ be a sequence of i.i.d. random variables with finite mean $\mu$. Let $S_n = \frac{1}{n}\sum_{i=1}^{n}X_i$

$$ S_n \rightarrow \mu \text{ as } n \rightarrow \infty $$

or formally, for any $\epsilon > 0$,

$$ \lim_{n\rightarrow\infty}P(|S_n-\mu|>=\epsilon) = 0 $$

This comes straight from the Chebyshev's inequality:

$$ P(|S_n-\mu|>=\epsilon) \leq \frac{\text{Var}(S_n)}{\epsilon^2} = \frac{\sigma^2}{n\epsilon^2} $$

and then take the limit

# Typicality

The **typical set** is a set of sequence whose probability is close to $2^{-nH(x)}$, where $n$ is the length of the sequence. It represents the most likely sequences given large $n$. 

To illustrate this, consider a Bernounlli rv $X \sim Ber(1/4)$, a sequence in the typical set is one whose **fraction of** $1$ equals $1/4$. In which case, the probability is given by $p^{np}(1-p)^{n(1-p)} = 2^{\log{p}np}2^{\log{(1-p)n(1-p)}} = 2^{-nH_2(p)}$

## Asymptotic Equipartition Property (AEP)

If $X_1, X_2, ... \text{are i.i.d} ~ P_X, \text{then for any }\epsilon > 0$

$$ \lim_{n\rightarrow\infty}Pr( |-\frac{1}{n}\log{P_X(X_1,X_2,...,X_n)-H(x)|<\epsilon) =  1}$$

This is simply the WLLN, with substituion of r.v $Y = -\log P_X(X)$. We note that $E[Y] = H(X)$

With this, we can formally define the typical set

The **typical set** $A_{\epsilon,n}$ with respect to $P$ is the set of sequences $(x_1,x_2,...,x_n)\in X^n$ for which:

$$ 2^{-n(H(X)+\epsilon)}\leq P(x_1,...,x_n) \leq 2^{-n(H(X)-\epsilon)}$$

In plain English: *Sequences whose probability is concentrated around* $2^{-n(H(X))}$

> The above definition should always be used to find a typical set. That is 1) you need to calculate the joint probability $P(X_1,X_2,...,X_n)$ in term of occurrences $n_{X_1},n_{X_2},...$ etc, and the entropy $H(X)$. Then equating $-\frac{1}{n}\log(P(X_1,X_2,...,X_n))=H(X)$. An example can be found on [Q1 Tripos 2019](Crib_3F7_2019.pdf)

## Properties of the Typical Set

1. When n is suffciently large, all sequences fall within the typical set
2. The size of the typical set is upper-bounded by $2^{n(H(X)+\epsilon)}$
3. For sufficiently large n, the size of the typical set is lower-bounded by $(1-\epsilon)2^{n(H(X)-\epsilon)}$

## Fundamental Limits for Compression

We can establish the fundamental limit of compression by using the typicality set. We know that as $n\rightarrow\infty$, all the samples will fall into the typical set with equal probability $2^{-nH(x)}$. The size of the typical set is given by $2^{nH(x)}$. Therefore, if we are to encode each sequence in bits, the least number of bits we need to use will be $nH(x)$ bits.

# Prefix-free codes and Kraft inequality

A prefix-free code refers to the fact that no codeword is a prefix of another codeword. Therefore, the sequence of codewords must be **uniquely decodable and encodable**. We are only interested in prefix-free code. The easiest way to see this is to look at a code tree, where we fix that *going left = 0* and *going right = 1*. The prefix-free code condition becomes **every codeword must be the leaf of the tree** 

![](fig1.png)

## Kraft inequality

Now the question of interests is, given a sequence of codeword length ${l_1, l_2, ..., l_n}$, does it exist a prefix-free coding scheme? This is useful because we often want to specify the codeword length for each symbol, without loss of *uniquely decodability* 

The Kraft inequality shows that, such coding scheme exists iff

$$\sum_{i=1}^{N}2^{-l_i} \leq 1$$

This can be easily proven if we establish a tree of length $l_{max} > \max(l_1,l_2,...,l_n)$, and view the placing of each codeword as *eliminating* $2^{l_{max}-l_i}$ leaves of the tree. We must make sure the leaves eliminated is less than $2^{l_{max}}$.

# Practical Coding

## Shanno-Fano Coding

The idea of Shanno-Fano coding comes from a simple observation. The fundamental limit for data compression states that the average code length cannot be smaller than the entropy. 

$$E(\frac{\sum_i l_i}{N}) >= H(X) = E(-\log{p(X_i)})$$

Hence, we can encode each symbol $x_i$ by a codeword of length $\lceil {log(1/p(x_i))} \rceil$. The smallest integer that is greater than the quantity. This is the *Shanno-Fano Coding*. We will now prove that the Shanno-Fano coding is a **prefix-free** code by design, and derive its **upperbound of bits/symbol**.

Let $X$ be the signal source taking values from the set $[x_1,x_2,...,x_n]$. Let $l_i$ be the codeword length for the symbol $x_i$. The Shanno-Fano code dictates that:

$$l_i = \lceil{\log{1/p(x_i)}}\rceil < \log{1/p(x_i)} + 1$$

We first examine whether it is a *prefix-free* code by checking the Kraft's inequality:

$$\sum_i 2^{-l_i} <= \sum_i 2^{\log{(p(x_i))}} = 1 $$

Hence the Kraft's inequality is satisfied.

The expected code length for a symbol is given by

$$E(l) = \sum_i p_il_i < \sum_i p_i[\log{(1/p(x_i))}+1]$$

$$E(l) < H(X) + 1$$

Hence we at most pay a penality of 1 bit to encode each symbol.

Note that the Shanno-Fano code is **not optimal**. This can be seen from the code tree below.

![](fig2.png)

We can strictly improve the codeword length by moving one of the codeword to the right-most leaf. Therefore it is not an optimal coding scheme.

## Huffman Coding

Huffman coding is an **optimal** coding scheme in terms of the expected bits/symbol. To achieve this, we recognize two facts about an optimal code:

1. If $p_i > p_j$, the $l_i >= l_j$
2. The two least probable symbol should have the same code length.

The Huffman encoding algorithm is as follows:

1. Take the two least probable symbols in the alphabet. These two symbols will be given the longest codeword with *equal* length. (i.e., only differ in the last digit)
2. Combine these symbols into a single symbol (probability is summed), and repeat.

![](fig3.png)

We now turn to showing that it is *prefix-free* and *optimal*. By design, we can see that it is prefix-free because codewords are only given at the leaves of the tree.

We can use induction method to prove its optimality. Assume we have the optimal encoding for $m$ symbols $C_m^*$, we can show that we can obtain an optimal encoding for $m-1$ symbols, merging the two least probable symbols to obtain $C_{m-1}$. The converse (from $m-1$ to $m$) can be shown as well. We then show that $C_{m-1}^* = C_{m-1}$

Huffman coding is optimal but it suffers from severe overhead when $H(X)$ is small, making the 1 bit penalty unacceptable. One way to work around this is to take the symbol block, e.g., $(X_1, X_2, ..., X_k)$ as a new signal to encode, which has a larger entropy. The expected code length per symbol becomes:

$$H(X) + \frac{1}{k}$$

If $k$ is large, then we should approach the limit. However, we need to note that the size of the outcome set increases **exponentially** as $|X|^k$, therefore Huffman coding is hard to scale to encode long sequences. The table of codeword can be very large.

## Arithmetic Coding

Haffman code suffers from bad generalization to code block and arithmetic coding is a better solution to encode long sequence. The idea is using the **infinity nature of the real axis**

We first encode our symbol in *intervals*. Suppose we have symbols $x_1,...,x_n$ with probability $p_1,...,p_n$, we divide the real axis into intervals of $[(0,p_1), (p_1,p_1+p_2),...,(1-p_n, p_n)]$

Next, we need to *encode* the intervals with binary representation. This is done by finding the **largest dyadic interval** $[i/2^l (i+1)/2^l]$ fully resides in the divided intervals, and then take the binary representation of the *lower end-point* as the encoding

The full procedure is given below:

![](fig4.png)

> Representing fraction in binary: adding each bit means take the lower (0) or upper (1) part after dividing the current interval equally. For example: 010 can be interpreted as [0,1] (010) -> [0,0.5] (10) -> (0.25, 0.5) (0) -> (0.25, 0.375)

### Expected Code Length and prefix-free

We have: $2^{-l_i} <= p_i$ **and** a extra bit to fit the dyadic interval (say, for the interval (0.01, 0.25), hence $l_i = \hat{\log{1/p_i}} + 1 <= \log{1/p_i} + 2$. It is not an optimal coding scheme for each bit

By design, the encoding is prefix-free. Suppose a code is the prefix of the other, then it must fully contain the other code, thus breaking the encoding scheme - **encode with largest dyadic interval fully resides in the interval**

### Encoding and Decoding

We do not encode the individual symbol but **encode the sequence**. We can encode a sequence with the following steps:

1. Compute the interval $(a, b)$ for the coded sequence recursively
2. Compute the probability of the sequence $p(s)$. We know the length of the code word is either $\log (\hat{1/p(s)})$ or $\log (\hat{1/p(s)})+1$
3. Compute $a\times 2^l$ and $b\times 2^l$ and check if the there exists two integers $i, i+1$ between them. If not, go for the longer codelength. We find the dyadic interval $[i/2^l, (i+1)/2^l]$
4. The codeword is the binary representation of $i$ with $l$ digits

![](fig7.png)

Here's an example:

![](fig5.png)

Decoding is straight forward as well:

![](fig6.png)



# Relative Entropy

Relative entropy can be seen as the **distance** between distribution, it is also called the *Kullback-Leiber (KL) Divergence* for that reason. Written as 

$$D(P||Q) = \sum_x P(x)\log{\frac{P(x)}{Q(x)}}$$

Note that in general, $D(P||Q)\neq D(Q||P)$. Hence it's not a true distance. That's why it's called divergence rather than distance.

Relative entropy is always **non-negative**. This can be proved by using IT inequality.

## Redundancy in Coding

We can interpret relative entropy as the redundant codeword length we use because we are encoding the source with $Q(x)$ rather than its true distribution $P(x)$

$$E(l) = \sum_x P(x)\log{1/Q(x)}$$

$$ \sum_x P(x)\log{\frac{P(x)}{P(x)Q(x)}} = H(x) + D(P||Q)$$

## Likelihood-ratio Thresholding

### Optimality (Neyman-Pearon lemma)

# Mutual Information

Mutual information is defined as 

$$I(x;y) = H(X)-H(X|Y) = H(Y)-H(Y|X)$$

This is best visualized by a Venn diagram

![](fig8.png)

In plain English, mutual information can be viewed as *the reduction* in the *uncertainty* of a random variable after observing another random variable. In this context, we can view the entropy as the *amount of information* in bits.

Mutual information can also be defined by *the KL-divergence of the joint pmf and the product of their marginals*.

$$I(x,y) = D(P_{XY}||P_XP_Y)$$

Following that we can see that the mutual information should be **strictly positive**

## Conditional Mutual Information

Given $X,Y,Z$ jointly distributed according to $P_{XYZ}$, the conditional mutual information $I(X; Y|Z)$ is defined as

$$I(X; Y|Z) = H(X|Z) - H(X|Y,Z) = H(Y|Z) - H(Y|X,Z)$$

This can be rewritten as the **chain rule of mutual information**:

$$I(X;Y,Z) = I(X;Z)+I(X;Y|Z)$$

> The above is best illustrated with Venn's diagram. where we can view `;` as intersect operator, `,` as union and `|` as exclude operatior.

![image-20210329001447898](image-20210329001447898.png)

$Z$ can be treated as the common *pre-condition*, the formular can be obtained by expanding without the condition and then conditioning every term with $Z$. In short, we only replace the entropy with conditional entropy.

# Discrete Memoryless Channel (DMC)

A discrete memoryless channel is defined by the input alphabet $X$, output alphabet $Y$, and the transitional probability $P_{Y|X}$ which represents the error probability given $x$.

A useful toy channel is called the noisy keyboard channel. 

![](fig9.png)

## Channel Capacity

The channel capacity is defined as

$$ C = \max_{P_x}I(x,y)$$

where $I(x,y)$ is the mutual information. We can write $I(x,y) = H(Y)-H(Y|X)$ because $P(Y|X)$ is given to define a channel. The mutual information can be seen as **a reduction in the uncertainty of Y given X**, which explains why this is the channel capacity.

## Channel Capacity for Typical Channels

Below is the calculation for typical channels. The general idea is expand $I(X;Y) = H(Y)-H(Y|X)$. $H(Y|X)$ is usually a nice entropy (usually of the form $H_2(p)$). $H(Y)$ may be obtained by $H(Y) \leq\log_2(|Y|)$, where $|Y|$ is the size of the constellation. In some cases (for example BEC), we need to obtain $P_Y(y)$ by feeding forward $P_X(x)$. Some results are:

1. **Noiseless Binary Channe**l:  $C = 1$ bit/transmission
2. **Binary Symmetric Channel** (BSC): $C = 1-H_2(p)$, where $p$ is the cross-over probability
3. **Noiseless Keyboard Channel**: $C = \log26-1$
4. **Binary Erasure Channel** (BEC): $C = 1-\epsilon$, where $\epsilon$ is the erasure probability

![image-20210329154828610](image-20210329154828610.png)

# The Channel Coding Theorem

## Definition of Channel Code

We use the channel $n$ times to transmit $k$ bits. We can think of each sequence of $k$ bits as indexing a message $W$ in the set ${1,2,...,2^k}$. Therefore, $k$ bits -> $2^k$ messages.

An $(n,k)$ channel code of rate $R$ of a channel $(X,Y,P_{Y|X})$ consists of:

1. A set of messages ${1,2,...,2^k}$
2. An **encoding** function $X^n : {1,2,...,2^k}\rightarrow X^n$ which assigns a *codeword* to each message. The set of codewords is called the *codebook*
3. A **decoding** function $g : Y^n \rightarrow {1,...,2^{nR}}$, which produces a guess of the transmitted message for each received error.

> $R$ is the rate of the channel and $R = \frac{k}{n}$

## Intuitive Rationale for Channel Capacity

Take the BSC(0.1) as an example. 

![](fig10.png)

The idea is that, if we can map the **non-intersecting jointly typical output sequence** back to their input, then with high probability we would have achieved transmission with zero error. The number of these non-intersecting output typical sets is *at maximum* given by $2^{nH(Y)} / 2^{nH(Y|X)} = 2^{nR}$. Hence we can see that $R = I(x;y) = H(Y)-H(Y|X)$. The same idea applies for generally DMCs.

## Joint Typical Set

The set $A_{\epsilon, n}$ of *jointly typical* sequence $(X^{(n)}, Y^{(n)})$ with respect to a joint pmf $P_{XY}$ is defined as 

$$ A_{\epsilon, n} = {(x^n,y^n) \in X^n \times Y^n} \text{such that} $$

$$ |-\frac{1}{n}\log P_X(x^n) - H(X)| < \epsilon $$

AND

$$ |-\frac{1}{n}\log P_Y(y^n) - H(Y)| < \epsilon $$

AND

$$ |-\frac{1}{n}\log P_{XY}(x^n, y^n) - H(X, Y)| < \epsilon $$

where $P_{XY}(x^n, y^n) = \prod_{i=1}^n P_{XY}(x_i,y_i)$

We can use the following square to illustrate the idea. The dots are the sequences which are jointly typical.

![](fig11.png)

### The Joint AEP

Let $(X^n, Y^n)$ be a sequence drawn i.i.d. according to $P_{XY}$, then we have:

1. $Pr\left((X^n, Y^n)\in A_{\epsilon, n})\right) \leftarrow 1 \text{ as } n \rightarrow \infty $
2. $|A_{\epsilon, n}| < 2^{n(H(X,Y)+\epsilon)}$
3. If $(\hat{X}^n, \hat{Y}^n)$ are drawn i.i.d. from their respective marginals. I.e., from $P(X)P(Y)$, then we have

$$Pr\left((\hat{X}^n, \hat{Y}^n) \in A_{\epsilon, n} \leq 2^{-n(I(X,Y)-3\epsilon})\right)$$

To prove the Joint AEP, (1) and (2) basically follows from uni-variate AEP. For (3), the square visualization is useful.

![](fig12.png)

## The Channel Coding Theorem

![](fig13.png)

We describe the probability of error of a code using the following two metrics:

1. The **maximal** probability of error of the code is defined by

$$\max_{j\in \{1,2,...,2^{nR}\}} Pr(\hat{W}\neq j | W = j)$$

2. The **average** probability of error of the code is defined by

$$\frac{1}{2^{nR}}\sum_{j=1}^{2^{nR}} Pr(\hat{W}\neq j | W = j)$$

We know turn to proving the channel coding theorem 

# Data-Processing and Fano's Inequality

![](fig15.png)

**Data Processing Inequality**

If $X-Y-Z$ form a Markov chain. That is, $P(Z|XY) = P(Z|Y)$, then $I(X;Y) \geq I(X;Z)$

![](fig16.png)

**Fano's Inequality**

For any estimator $\hat{X}$ such that $X - Y - \hat{X}$ forms a Markov chain, the probability of error $P_e = Pr(\hat{X}\neq X)$ satisfies

$$1 + P_e\log |X| \geq H(X|\hat{X}) \geq H(X|Y)$$

or

$$P_e \geq \frac{H(X|Y)-1}{\log |X|}$$

# Differential Entropy and AWGN

## The additive white Gaussian noise (AWGN) channel

The definition of AWGN is given below:

![](fig14.png)

We have three assumptions on the channel:

1. Input $X(t)$ is *power-limited* to $P$. That is, average power over time $T$ must $\leq P$ for large $T$
2. $X(t)$ is *band-limited* to $W$. $\rightarrow$ Fourier transform of $X(t)$ must be zero outside $[-W, W]$
3. Noise $N(t)$ is a random process assumed to be *white* Gaussian.

With power and bandwidth constraint we can show that the **discrete-time** system is equivalent to the continuous channel, where

$$Y_k = X_k + Z_k$$

Power constraint: $\frac{1}{n} \sum_{k=1}^n X_k^2 \leq P$

White noise: $Z_k$ are i.i.d. Gaussian with mean $0$, variance $\sigma^2$

We want to compute the capacity of this discrete-time AWGN channel in bits/transmission.

## Differential entropy

Differential entropy is defined for continuous random variable $X$ with pdf $f_X$ as such:

$$h(X) = \int_{-\infty}^\infty f_X(u)\log \frac{1}{f_X(u)}du$$

Consider a uniform distribution in the interval $[0, a]$, we can show that $h(X) = \log a$. We note that if $a < 1$ the differential entropy is negative!

The correct interpretation for differential entropy is **the uncertainty relative to that of a Uni[0,1]**. In other words, $Uni[0,1]$ is the baseline which has $0$ differential entropy.

The differential entropy for a Gaussian random variable is 

$$h(X) = \frac{1}{2} \log (2\pi e\sigma^2)$$

### Properties of differential entropy

In brief, the equality of joint entropy, conditional entropy, mutual information and chain rules are identical to the discrtete case. The main difference is that the **differential entropy can be negative**. Mutual information and KL-divergence are non-negative as well.

## Capacity of AWGN channel

The capacity is given by $\max I(X;Y)$ as before. 

$$I(X;Y) = h(Y) - h(Y|X) = h(Y) - h(X+Z|X) = h(Y) - h(Z)$$

where $ Z \sim N(0,\sigma^2)$, hence $h(Z) = \frac{1}{2} \log (2\pi e\sigma^2)$

We note that $E[Y^2] = E[X^2] + \sigma^2$, since $X$ is power limited, we have $E[Y^2] \leq P + \sigma^2$

> Among all random variables $Y$ with $E[Y^2] \leq P+\sigma^2$, the one with the maximum differential entropy is $Y \sim N(0, P+\sigma^2)$, a Gaussian distributio. To show this, we let $f$ be the density function and $\phi$ be a Gaussian density function. We prove the result by remembering $D(f||\phi)\geq0$

We proceed with 

$$I(X;Y) = \frac{1}{2}2\pi e(P+\sigma^2) - \frac{1}{2}2\pi e\sigma^2 = \frac{1}{2}\left( 1 + \frac{P}{\sigma^2}\right)$$

Therefore, the capacity of the AWGN channel is given by:

$$C = \frac{1}{2}\log \left( 1 + \frac{P}{\sigma^2}\right)$$

$C$ depends only on the *signal-to-noise* ratio (snr) $\frac{P}{\sigma^2}$

If the channel has bandwidth $W$, it can be used $2W$ times per second. Therefore, the capacity in bits/second is 

$$W\cdot \log \left( 1 + \frac{P}{\sigma^2}\right) \text{bits/sec}$$

# Binary Linear Block Codes

![Screen Shot 2021-01-21 at 15.47.32](Screen Shot 2021-01-21 at 15.47.32.png)

We now turn our focus on constructing good and practical binary codes, particularly for the binary symmetric channel and the binary erasure channel.

## Block Code

An $(n,k)$ binary block code maps every block of $k$ data bits into a length $n$ binary codeword. The **rate** $R=k/n$ 

> Criterion for good code:
>
> 1. Rate R as high as possible (close to channel capacity)
> 2. Low probability of decoding error
> 3. Computationally efficient to encode and decode

The error correction capability of any $(n,k)$ block code depends on the *pairwise distances* between the codewords.

## Hamming Distance

The *Hamming distance* $d(x,y)$ between two binary sequences $x,y$ of length $n$ is the number of positions in which $x$ and $y$ differ.

Let $\mathcal{B}$ be a code with codewords ${c_1,...,c_M}$. Then the minimum distance $d_{min}$ is the smallest Hamming distance between any pair of codewords. That is,

$$d_{min} = \min_{i\neq j}d(c_i, c_j)$$ 

## Optimal Decoding of Block Code

Given a codebook $\mathcal{B}$ with $M$ codewords, the optimal decoder is one that minimizes the probability of decoding error. For any channel described by $P_{Y|X}$, the the input bits/messages corresponding to the codewords are equally likely, then the optimal decoder given the received length-n sequence $y$ is the max-likelihood decoder given by

$$\hat c = \arg\max_c Pr(y|c)$$

![Screen Shot 2021-01-21 at 16.08.06](Screen Shot 2021-01-21 at 16.08.06.png)

We can succcessfully **correct** any pattern of $t$ errors if $t\leq \lfloor\frac{d_{min}-1}{2}\rfloor$.

## Linear Block Codes (LBC)

A $(n,k)$ linear block code (LBC) is defined in terms of $k$ length-$n$ binary vectors, say $\vec{g_1},...,\vec{g_k}$. A sequence of $k$ data bits, say, $x = (x_1,...,x_k)$ is mapped to a length-$n$ codeword $c$ as follows.

$$c = x_1\vec{g_1}+x_2\vec{g_2}+...+x_k\vec{g_k}$$

> A direct consequence is if $c_1$ and $c_2$ are in the codebook then $c_1+c_2$ is a valid code as well.

![Screen Shot 2021-01-21 at 16.13.13](Screen Shot 2021-01-21 at 16.13.13.png)

- The $k\times n$ matrix $G$ is called a **generator matrix** of the code
- $k$ is called the code **dimension**, $n$ is the **block length**
- We assume vectors are *row* vectors

> The generator matrix for a set of codewords is not unique, row exchange doesn't change the row space

### Systematic Generator Matrices

Among all possible generator matrices for a code, we often prefer one that is of the form

$$G = [I_k | P]$$

where $I_k$ is the $k \times  k$ identity matrix and $P$ is a $k\times (n-k)$ matrix. Such $G$ is called a **systematic** generator matrix. The trailing $(n-k)$ bits are called *pairity bits*. 

Given any generator matrix, we can find a systematic version:

1. Use **elementary row operations**: swap rows; replace any row by a sum of that row and any other rows.

   If G1 is obtained from G2 via elementary row operations, then G1,G2 have the same set of codewords; only the mappings are different.

2. To bring G1 into systematic form, may also need to **swap columns**. This leads to rearranging the components of the codewords.

### LBCs as Subspaces

Let $C$ be an $(n,k)$ LBC with codewords ${c_0, . . . , c_{M−1}}$. $C$ is a subspace of $\{0,1\}^n$.

> A *subspace* means it is closed under vector addition and scaler multiplication

**Properties of codewords**

1. The code is a k-dimensional subspace of vectors from $\{0,1\}^n$
2. The rows of $G$ forms a basis for $C$. Each codeword is a linear combination of basis vectors
3. The sum of any two codewords is also a codeword; the all-zero vector 0 is always a codeword

### The Parity Check Matrix 

The orthogonal complement of $C$, denoted $C'$ is defined as the set of all vectors in $\{0,1\}^n$ that are orthogonal to each vector in $C$.

- $C'$ is a subspace

- $C'$ has dimension $n-k$ 

- The $(n-k)\times n$ matrix $H$ is called the **parity check matrix**

- Each codeword $c\in \mathcal{C}$ is orthogonal to each row of $H$.

  $$cH^T = 0 \rightarrow xGH^T = 0\text{ for all x }\rightarrow GH^T = 0$$

If we have systematic $G = [I_k | P]$, then take

$$H = [P^T | I_{n-k}]$$

The *parity-check equations* of the code is obtained from $cH^T=0$

> The parity check matrix $H$ is a complete description of linear code, just like the generator matrix $G$

### Minimum Distance of an LBC

The Hamming distance between two binary vectors $u, v$ can be expressed as $$d(u, v) = wt(u + v)$$,
where $wt$ refers the Hamming weight of the vector, i.e., the **number of ones** in it.

Recall that the minimum distance of a code is

$$d_{min} =\min d(c_i,c_j)=\min_{i\neq j} wt(c_i +c_j) $$

For any two codewords $c_i,c_j $of an LBC, note that $c_i +c_j$ is also a codeword $c_k$Therefore

$$d_{min} = \min wt (c_i + c_j ) = \min_{c_k\neq 0} wt (c_k )$$

In short, The minimum distance of an LBC equals **the minimal Hamming weight among the non-zero codewords**

Let $H$ be the parity check matrix of code $\mathcal{C}$, The minimal distance of $C$ is the **smallest number of columns of $H$ that sum to 0**.

> $d_{min}$ gives you the **guranteed** error correction capability. However, to design practical code, it's often the case that we want *small error probability* but not zero error. Hence a length 10000 codeword transmitted over BSC(0.1) may have $d_{min}$ =50 rather than 2000

# Low Density Parity Check (LDPC) Codes for the Binary Erasure Channel

## The Binary Erasure Channel (BEC)

![Screen Shot 2021-01-22 at 15.30.02](Screen Shot 2021-01-22 at 15.30.02.png)

## Decoding

### Matrix inversion method

Given a parity check matrix $H$ and a received $y$, we can try to recover the erased bits by the "matrix inversion" method. That is, since $cH^T$=0, we can solve the system of linear equations to recover the bits. This is typically done with "Gaussian elimination".

Note that this can be computationally expensive, with complexity at $n^3$

### Iterative Decoding

We use an example to illustrate the process. 

![Screen Shot 2021-01-22 at 15.36.48](Screen Shot 2021-01-22 at 15.36.48.png)

![Screen Shot 2021-01-22 at 15.37.56](Screen Shot 2021-01-22 at 15.37.56.png)

Repeat the process utill all the erased bits are recovered.

Iterative decoding is possible whenever the effective matrix can be triangulated by just row swaps, i.e., no linear combinations of rows needed for Gaussian elimination. For this to happen for most erasure patterns, the parity check matrix needs to have a **low density of ones**.

## Low Density Parity Check (LDPC) Matrix

### Regular LDPC codes

![Screen Shot 2021-01-22 at 15.42.09](Screen Shot 2021-01-22 at 15.42.09.png)

$d_v$ is the column weight and denotes how many equations a bit is involved in. $d_c$ is the row weight denoting how many code bits a parity check equation involves.

The number of 1's in a regular parity check matrix is given by $d_v n=d_c(n-k)$

The *design rate* of a regular LDPC code is $\frac{k}{n}=1-\frac{d_v}{d_c}$

### Factor graph of a linear code

It's useful to represent the parity check matrix as a *factor graph* to analyse the iterative decoder and design good LDPC matrices.

![Screen Shot 2021-01-22 at 15.53.31](Screen Shot 2021-01-22 at 15.53.31.png)

- Each column $j$ of $H$ is a variable node $v_j$ (represented by circles)
- Each row $i$ of $H$ is a check/constraint node $c_i$ (represented by square)
- A one in entry $(i,j)$ of $H$ means that variable node $j$ is connected to check node $i$, i.e., code bit $j$ is involved in the parity check equation descrived by the $i$th row.

### Iterative Decoding as Message Passing

![Screen Shot 2021-01-22 at 15.58.06](Screen Shot 2021-01-22 at 15.58.06.png)

The idea is that messages are sent between the checks and variables. The message from check $c$ to variable $v$,  $m_{cv}$ is a function of the messages $\{m_{v'c} \}$ coming into the check (exclusing $v$ itself). And vice versa.

The message passing rules for decoding on the BEC are as follows:

![Screen Shot 2021-01-22 at 16.06.17](Screen Shot 2021-01-22 at 16.06.17.png)

Iterative decoding is carried out becuase if there is only one unknown variables among all connected to a check, the check can recover the bit and send it back to the variable using the above scheme. This only works when there is only one erasure among all connections to a check, which again verifies that low density indeed fascilitates iterative decoding.

The complexity of the message passing decode is $\propto (\text{# of edges in graph})(\text{# of iteration})$

## Designing goood LDPC codes

The standard way to construct an LDPC code is to: 

- first choose a *degree distribution* that specifies the distribution of weights on the nodes/edges of the graph.
- Then fix a (large) code length $n$, and pick an $H$ with this degree distribution, either at random or through some determinisic construction

### Degree distributions

![Screen Shot 2021-01-22 at 16.19.35](Screen Shot 2021-01-22 at 16.19.35.png)

Let $L_i$ be the fraction of left (variable) nodes of degree $i$, i.e., the fraction of columns in $H$ with weight $i$.

Let $R_i$ be the fraction of right (check) nodes of degree $i$, i.e., the fraction of rows in $H$ with weight $i$.

The degree distributions of the variable and check nodes are often written in terms of the following "**node-perspective**" polynomials:

$$L(x) = \sum^{d_{v,max}}_{i=1}L_ix^i, R(x)=\sum^{d_{c,max}}_{i=1}R_ix^i $$

For the above graph, we have

$$L(x) = \frac{3}{5}x^2+\frac{3}{10}x^3+\frac{1}{10}x^4, R(x)=\frac{1}{5}x^4+\frac{3}{5}x^5+\frac{1}{5}x^6$$

The average degree of a variable node is $\bar d_v = L'(1)$

The average degree of a check node is $\bar d_c = R'(1)$

The total number of edges in the graph (number of ones in $H$) is $\bar d_vn = \bar d_c(n-k)$

> Hence the design rate of the code is $ 1 - \frac{\bar d_v}{\bar d_c}$

We can also define degree distributions from the edge perspective:

$λ_i$: Fraction of edges connected to variable nodes of degree i,  i.e., the fraction of ones in H in columns of weight i.

$ρ_i$: Fraction of edges connected to check nodes of degree i, i.e., the fraction of ones in H in rows of weight i.

$$\lambda(x) = \sum^{d_{v,max}}_{i=1}\lambda_ix^{i-1}, \rho(x)=\sum^{d_{c,max}}_{i=1}\rho_ix^{i-1} $$

![Screen Shot 2021-01-22 at 16.30.15](Screen Shot 2021-01-22 at 16.30.15.png)

![Screen Shot 2021-01-22 at 16.30.50](Screen Shot 2021-01-22 at 16.30.50.png)

### Density Evolution

Density evolution is a technique to predict the decoding performance of codes with a given $\lambda(x)$ and $\rho(x)$ for large $n$. 

- Let $p_t$ denote the probability that an outgoing $v → c$ message (along an edge picked uniformly at random) is an erasure (“?”) in step t.

  On a BEC with erasure probability ε, for t = 0 we have $p_0 = ε$.

- Let $q_t$ denote the probability that an outgoing c → v message is a “?” in step t. We start with $q_0=1$ as the first $c\rightarrow v$ messages are all erasures.

![Screen Shot 2021-01-22 at 16.37.06](Screen Shot 2021-01-22 at 16.37.06.png)

![Screen Shot 2021-01-22 at 16.39.06](Screen Shot 2021-01-22 at 16.39.06.png)

### Density Evolution for Irregular Codes

![Screen Shot 2021-01-23 at 13.39.54](Screen Shot 2021-01-23 at 13.39.54.png)

Following the same analysis, we note that $p_t$ and $q_t$ can be expressed in terms of $\lambda(\cdot)$ and $\rho(\cdot)$by definition. We have the following evolution equation:

$$p_t = \epsilon\lambda(1-\rho(1-p_{t-1}))$$

# Low Density Parity Check (LDPC) codes for general binary input channels

We study LDPC codes for general *binary-input* channels. The output alphabet can be arbitary, but we assume that the channel is symmetric.

Our framework covers the following three channels:

1. *Binary Erasure Channel* (BEC)

2. *Binary Symmetric Channel* (BSC)

3. Binary Additive White Gaussian Noise Channel (B-AWGN)

   $Y=X+N$, where the input $X\in\{+1, -1\}$ and $N\sim\mathcal{N}(0,\sigma^2)$ is additive white Gaussian noise. The channel input is generated from a binary (0/1) codeword as follows: *map each 0 code-bit to $X=+1$, and each 1 code-bit to $X=-1$*.

## The Set-up

![Screen Shot 2021-01-23 at 13.50.32](Screen Shot 2021-01-23 at 13.50.32.png)

## Message Passing Decoding (Belief Propagation)

The message is the desired bit-wise *a posteriori probabilities* (APPs). We will index the variable nodes by $j$ and the check nodes by $i$

In each iteration, the message passing decodes computes:

1) *Variable-to-check messages*

- Each v-node $j$ sends a message $m_{ji}$ to each c-node $i$ that it is connected to
- $m_{ji}(0)$ **is an updated estimate of the posterior probability (or belief) that the code bit $c_j=0$**
- $m_{ji}$ is computed using the channel evidence $P(c_j|y_j)$ and all the incoming messages into $j$ *except* from c-node $i$.

2) *Check-to-variable* messages

- Each c-node $i$ sends a message $m_{ij}$ to each v-node $j$ that it is connected to
- $m_{ij}(0)$ **is an updated estimate of the probability that the parity check equation $i$ is satisfied when $c_j=0$** 
- $m_{ij}$ is computed using all the incoming messages into $i$ *except* from v-node $j$

Using the assumption that the incoming messages at each node are *independent*, we now derive the message updates.

### Variable-to-check messages

![Screen Shot 2021-01-23 at 23.51.40](Screen Shot 2021-01-23 at 23.51.40.png)

$$m_{ji}(0)=P(c_j=0|y_j, \text{check }i_1, i_2,i_3 \text{ are all satisfied})$$

$$m_{ji}(1)=P(c_j=1|y_j, \text{check }i_1, i_2,i_3 \text{ are all satisfied})$$

Using Bayes theorem (and assuming independence of incoming messages):

$$m_{ji}(0)=\frac{a_0}{a_0+a_1}, m_{ji}(1)=\frac{a_1}{a_0+a_1}$$

where

$$a_0 = Pr(c_j=0, \text{check }i_1, i_2,i_3 \text{ are all satisfied }y_j)=P(c_j=0|y_j)m_{i_1j}(0)m_{i_2j}(0)m_{i_3j}(0)$$

$$a_1 = Pr(c_j=1, \text{check }i_1, i_2,i_3 \text{ are all satisfied }y_j) = P(c_j=1|y_j)m_{i_1j}(1)m_{i_2j}(1)m_{i_3j}(1)$$

In general, $m_{ji}$, the outgoing message from each v-node $j$ to c-node $i$ is computed as follows:

In iteration 1, $m_{ji}(0)=P(c_j=0|y_j)$

For $t>1$:

$$m_{ji}(0)=\frac{P(c_j=0|y_j)\prod_{i' \text{\\}i}m_{i'j}(0)}{P(c_j=0|y_j)\prod_{i' \text{\\}i}m_{i'j}(0)+P(c_j=1|y_j)\prod_{i' \text{\\}i}m_{i'j}(1)}$$

$$m_{ji}(1)=1-m_{ji}(0)$$

Here the notation $i'\text{\\}i$ denotes all the c-nodes $i'$ connected to $j$ except $i$

> The meesage $m_{ji}(0)$ thus is an updated probability ('belief') that $c_j=0$ based on the incoming messages $m_{i'j}$. The assumption is that the incoming messages are independent.

### Check-to-variable messages

We first provide a lemma:

> Consider a sequence of $M$ independent binary digits $b_1,...,b_M$ such that $P(b_k=1)=p_k$ for all $k$. Then the probability that the sequence $(b_1,...,b_M)$ contains an even number of ones is 
>
> $$\frac{1}{2}+\frac{1}{2}\prod_{k=1}^M(1-2p_k)$$

Using this result, we have

$$m_{ij}(0) =\frac{1}{2}+\frac{1}{2}\prod_{j'\text{\\}j}(1-2m_{j'i}(1))$$

this is because a even number of bits will sum to $0$ for the check bit.

In summary:

![Screen Shot 2021-01-24 at 00.17.46](Screen Shot 2021-01-24 at 00.17.46.png)

### Channel Evidence $P(c_j|y_j)$

The channel evidence can be computed using Bayes rule and the channel transition probabilities:

$$P(c_j|y_j)=\frac{P(c_j)P(y_j|c_j)}{P(y_j)}$$

![Screen Shot 2021-01-24 at 00.25.31](Screen Shot 2021-01-24 at 00.25.31.png)

![Screen Shot 2021-01-24 at 00.25.44](Screen Shot 2021-01-24 at 00.25.44.png)

### Log domain message passing

The messages in the above decoding algorithm involve multiplying lots of probablities, which can cause the implementation to be numerically unstable. Hence belief propagation decoding is usually implemented with *log-likelihood ratios* (LLRs), which turns most of the multiplications into additions.

We now have:

$$L_{ji}=\ln\frac{m_{ji}(0)}{m_{ji}(1)}, L_{ij}=\ln\frac{m_{ij}(0)}{m_{ij}(1)}$$

The algorithm then becomes 

![Screen Shot 2021-01-24 at 00.39.44](Screen Shot 2021-01-24 at 00.39.44.png)

### Algorithm termination

The algorithm is run for a *pre-determined number of steps*, and the final LLRs for each code bit are computed as

$$L_j = L(y_j)+\sum_{i'}L_{i'j} \text{ for }j=1,...,n$$

The final decoded codeword is $\hat c=(\hat c_1,...,\hat c_n)$ where $\hat c_j = 0 \text{ if } L_j \geq 0, 1 $ otherwise