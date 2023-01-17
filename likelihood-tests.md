---
jupytext:
  cell_metadata_filter: -all
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---
# Likelihood tests

## Receiver operating characteristic (ROC) curves

Receiver operating characteristic (ROC) curves help use to further understand the competing objects true positive rate (TPR) and false positive rate (FPR). While we want TPR to be as large as possible, FPR on the other hand should be as small as possible. To find the balance between the TPR and FPR we use risk minimization:

$$
    R[\hat{Y}] := \mathbb{E} [\mathrm{loss}(\hat{Y}(X), Y)] = \alpha \mathrm{FPR} - \beta \mathrm{TPR} + \gamma
$$

with the assumption that $\alpha$ and $\beta$ are nonnegative and $\gamma$ is a constant and for all $\alpha, \beta, \gamma$ the risk-minimizing predictor is a likelihood ration test (LRT).

The question we are trying to answer with the help of ROC curves is if we can achieve any combination of FPR and TPR? 

<!-- Most of this information is from the lecture notebook and PPA Chapter 2 and therefore sounds very similar. I want to add some more context but I don't know exactly what with out adding something unneccessary 
-->
First let us take a look what ROC curves actually are. ROC curves are an intrinsic property of the joint distribution $(X, Y)$ defined as follows: For every $FPR \in [0, 1]$ it shows the best $TPR$ that can be achieved with any predictor with that $FPR$, resulting in a curve that in the $FPR$-$TPR$ plane. The curve shows the maximal $TPR$ for any given $FPR$. Constant predictors that either reject or accept all inputs are always shown on the ROC curve and that is why $(0,0)$ and $(1,1)$ are always on the ROC curve.

```{figure} ./images/roc_curve.png
Example of a generic ROC curve. source: [PPA Chapter 2](https://mlstory.org/)
```

### The Neyman-Pearson Lemma
To show how the ROC curve relates to the likelihood ratio tests we use the Neyman-Pearson Lemma. To show that we want to maximize the true positive rate (TPR) while applying an upper bound on the false positive rate (FPR). That gives us the following optimization problem we want to solve:

$$
    \begin{aligned}
        \text{maximize}  ~  &\mathrm{TPR} \\
        \text{subject to} ~ &\mathrm{FPR} \leq \phi
    \end{aligned}
$$

<!-- Is there any better way to highlight the lemma? -->
**Neyman-Pearson Lemma**
: Suppose the likelihood functions $p(x|y)$ are continuous. Then the optimal probabilistic predictor that maximizes $TPR$ with an upper bound on $FPR$ is a deterministic likelihood ratio test.

From the Lemma we can derive that the geometric properties of the ROC curve. It is traced out by the varying threshold in the like likelihood ratio test (LRT) from $-\infty$ to $\infty$.

### some properties of ROC curves
One property we already mentioned is that the points $(0,0)$ and $(1,1)$ are on the ROC curve as the cases where the constant predictor 0 for $(0,0)$ and 1 for $(1,1)$. In the LRT that means that the threshold for the point $(0,0)$ is $\infty$ and for the point $(1,1)$ is $0$.

Another property of the ROC curve is that it must lie above the main diagonal. We can see that for any $\alpha >0$. We can achieve $TPR = FPR = \alpha$ but because of the LRT in the **Neyman-Pearson Lemma** we have $FPR \le \alpha$ and therefore $TPR \ge \alpha$ and never $TPR \le \alpha$.

For any achievable for any achievable $\left(\operatorname{FPR}\left(\eta_1\right), \operatorname{TPR}\left(\eta_1\right)\right)$ and $\left(\operatorname{FPR}\left(\eta_2\right), \operatorname{TPR}\left(\eta_2\right)\right)$, the following is also achievable:

$$
    \left(t \operatorname{FPR}\left(\eta_1\right)+(1-t) \operatorname{FPR}\left(\eta_2\right), t \operatorname{TPR}\left(\eta_1\right)+(1-t) \operatorname{TPR}\left(\eta_2\right)\right)
$$

In conclusion we have another property of the ROC curve: The ROC curve is concave.

<!-- There is some code of the Snail example in the notebook here for an interactive plot that does not really work in the jupyter-book. Do you have an idea what I could put here instead?-->

<!-- I was thinking of adding a section for AUC but it was not really covered in the lexture so I decided against it.-->

## Maximum a posteriori and maximum likelihood
Something that is often said in statistical decision theory and that we also already noticed in the things that we have done: essentially all optimal rules are equivalent to likelihood ratio tests (LRTs). This isn't 100% true but it is true in most of the cases and mainly in the many very important prediction rules. The same thing is true for the maximum a posteriori (MAP) rule and the maximum likelihood (ML) rule that we want to take a look at now. 
We define the expected error of a predictor $\hat{Y}$ as the expected number of mistakes in classification. For example, if we predict $\hat{Y} = 1$ when $Y = 0$ is the actually true. The error is defined by the risk with the following cost: 

| loss | $\hat{Y}$ = 0 | $\hat{Y}$ = 1|
|-----|--------:|--------:|
| $Y$ = 0 | 0   |  1  |
| $Y$ = 1 | 1 | 0  |

Minimizing the risk with the defined cost is equivalent to minimizing the expected error and are also given by the likelihood rats tests. In this case we have: 

$$
    \frac{p_0(\operatorname{loss}(1,0)-\operatorname{loss}(0,0))}{p_1(\operatorname{loss}(0,1)-\operatorname{loss}(1,1))} = \frac{p_0}{p_1} \cdot \frac{1 - 0}{1 - 0} = \frac{p_0}{p_1}
$$
This results in the maximum a posteriori (MAP) rule: 

$$
    \begin{aligned}
    \hat{Y}(x) 
    &= \mathbb{1} \left\{ \mathcal{L}(x) \geq \frac{p_0}{p_1} \right\} \\
    &= \mathbb{1} \left\{  \frac{p(x \mid y = 1)}{p(x \mid y = 0)} \geq \frac{p_0}{p_1} \right\} \\
    &= \mathbb{1} \bigg\{  p_1 p(x \mid y = 1) \geq p_0 p(x \mid y = 0) \bigg\} \\
    &= \mathbb{1} \bigg\{  \mathbb{P}[Y = 1 \mid X = x] \geq \mathbb{P}[Y = 0 \mid X = x] \bigg\} \\
    &= \arg\max_{y \in \{ 0, 1\}} \mathbb{P}[Y = y \mid X = x]
    \end{aligned}
$$

The name comes from the expression $\mathbb{P}[Y = y \mid X = x]$ which called the posterior probability of $Y$ given $X$.

Equivalent to the MAP rule is the maximum likelihood (ML) rule when $p_0 = p_1$. The only difference is that we  use the likelihood of the point $x$ given $Y = y$ defined as $p(x | Y = y)$ instead of the posterior probability. The ML rule is defined as:

$$
    \hat{Y}_\text{ML}(x) = \arg\max_y p(x | Y = y)
$$

Like MAP the ML is also a likelihood ratio test, which is not a coincidence because Likelihood ratio tests are in most castes the optimal solution for optimization-driven decision problems.