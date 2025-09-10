# Biologically Plausible Learning for Medical AI

This repository accompanies our study, [Investigating the Application of Feedback Alignment to Medical AI]().

This work explores biologically plausible alternatives to backpropagation (BP), specifically Direct Feedback Alignment (DFA), tested in the context of medical machine learning on the image medical dataset for classification, PatchCamelyon (PCam), using **Vision Transformers (ViTs)** by introducing **ViT-DFA**, where standard gradient-based updates are replaced with DFA-based credit assignment.

---

## Objectives

1. Evaluate the performance of backpropagation vs. feedback-alignment-based learning in a ViT.
2. Investigate generalisation / robustness, convergence behaviour, and accuracy.
3. Explore learning dynamics and feature attribution of biologically plausible learning algorithms in medical AI.


---

## Core Principle

### 1. Standard Backpropagation (BP)

Let a feedforward network have $L$ layers, activations, $a^l = \phi(W^{l-1}a^{l-1} + b^l)$ , and loss, $\mathcal{L}(y, \hat{y})$.  

The gradient update is:

$$
\delta^L = \nabla_{\hat{y}} \mathcal{L} \odot \phi'(z^L), \quad
\delta^l = \big(W^{l+1}\big)^\top \delta^{l+1} \odot \phi'(z^l)
$$

$$
W^l \gets W^l - \eta \, \delta^l (a^{l-1})^\top
$$

Error signals are propagated back through the transpose of forward weights, ensuring exact gradient flow and error signals.

---

### 2. Direct Feedback Alignment (DFA)

Based on [Lillicrap et al. (2016), Nøkland (2016)].

Introduces biologically plausible credit assignment by bypassing symmetric weight transport.

Error at the output layer is `directly` projected backwards using fixed random feedback matrices, $B^l$, instead of $(W^{l+1})^T$ :

$$
\delta^l = B^l \delta^L \odot \phi'(z^l)
$$

This breaks symmetry by giving random signals but is observed to still lead to learning due to gradual alignment between $B^l$ and $W^l$.

---

| Feature                 | Backpropagation            | Feedback Alignment            |
| ----------------------- | -------------------------- | ----------------------------- |
| Gradient flow           | Exact                      | Approximate, random-projected |
| Weight symmetry         | Required (transpose usage) | Not required                  |
| Biological plausibility | Low; brain doesn't know forward weights | Higher                        |
| Convergence speed       | Typically faster           | Slower, depends on alignment  |

---

## Vision Transformer with DFA (ViT-DFA)

In addition to MLP experiments, we introduce **ViT-DFA**, which applies Direct Feedback Alignment to Vision Transformers:

- **Architecture:** Based on ViT-Tiny (`timm` implementation) with 12 transformer blocks, embedding dimension 192, patch size 16.
- **Modification:** The final block is replaced with a **custom DFA-enabled block** where error signals from the classification head are projected back using **random feedback matrices** instead of standard backprop gradients.
- **Backward Pass:** Implemented using a custom `torch.autograd.Function`, ensuring DFA replaces gradient flow while maintaining compatibility with PyTorch training loops.
- **Task:** Applied to **PatchCamelyon (PCam)**, a benchmark medical histopathology dataset, to evaluate biologically plausible learning in high-dimensional, image-based settings.

### Empirical Observations (ViT-DFA)

1. **Backpropagation (BP-ViT):** Achieves higher accuracy and faster convergence, as expected from exact gradient optimisation.
2. **ViT-DFA:**  
   - Converges slower but still learns meaningful representations.  
   - Shows reduced data efficiency compared to BP.  
   - Provides a **proof-of-concept** that biologically plausible DFA can scale beyond MLPs to transformer-based architectures.
3. **Key Insight:** Random feedback alignment in ViTs can still produce non-trivial learning, supporting claims that DFA may generalise in **deeper and more structured models**.

---

## Repository Structure

- `ViT_DFA.ipynb` – Core training and learning dynamics comparison for ViT_DFA.
- `DFA_model/` – Implementation of ViT with DFA-based backward pass.
  - `DFA_for_MLP.py` – Custom custom DFA-enabled block.
- `utits/` - Additional files for dataset and visualisation
  - `dataset.py` – Setting up PatchCamelyon dataset.
- `results/` – Contains logs, curves, and comparison plots.

---

## Conclusion

- **MLP-DFA:** Demonstrates biologically plausible alternatives for tabular medical AI.  
- **ViT-DFA:** First step towards applying biologically plausible learning principles to **transformer-based vision models** in medical AI.  
- **Future Work:** Scaling DFA to larger transformer models, hybrid biologically-inspired credit assignment, and applications in neuromorphic computing.

---
