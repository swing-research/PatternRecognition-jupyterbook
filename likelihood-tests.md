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
First let us take a look what ROC curves actually are. ROC curves are an intrinsic property of the joint distribution $(X, Y)$ defined as follows: For every $FPR \in [0, 1]$ it shows the best $TPR$ that can be achieved with any predictor with that $FPR$, resulting in a curve that in the FPR-TPR plane. The curve shows the maximal TPR for any given FPR. Constant predictors that either reject or accept all inputs are always shown on the ROC curve and that is why $(0,0)$ and $(1,1)$ are always on the ROC curve.

```{figure} ./images/roc_curve.png
Example of a generic ROC curve. source: [PPA Chapter 2](https://mlstory.org/)
```

### The Neyman-Pearson Lemma

### some properties of ROC curves

## Maximum a posteriori and maximum likelihood
