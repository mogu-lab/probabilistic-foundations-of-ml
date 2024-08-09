
# Probabilistic Foundations of Machine Learning (CS349), Fall 2024

## TODOs

1. Improve explanation of CDF
2. Add exercises for Bayesian prior + posterior unit
3. Add tutorial on calculus to skills check



## Installation

1. Install `micromamba` with `brew`. Initialize the shell with: `micromamba shell init -s zsh`.

2. Create the environment:
   ```
   mm create -n book -c conda-forge python=3.10 -y
   pip install -r requirements.txt
   ```
3. Activate environment: `mm activate book`

**Note:** Only install packages with `pip`. Micromamba seems to store the local paths on the machine.


## Embedding Canva Figures

1. Make figure public for viewing
2. Get embed code and extract URL from it
3. Paste the following in markdown with the extracted URL:
```
<div class="canva-centered-embedding">
  <div class="canva-iframe-container">
    <iframe loading="lazy" class="canva-iframe"
      src="URL">
    </iframe>
  </div>
</div>
```


## DeepNote Setup

1. Add student to workspace as editor
2. Ask student to duplicate homework project as private
3. Rename project "Homework N: FIRST-NAME LAST-NAME"
4. Click share, and invite `yy109@wellesley.edu` with edit access
5. To download a PDF version of your notebook, follow [these instructions](https://deepnote.com/docs/export-pdf)


## Gradescope

1. Give students course code (from course website)
2. sign in with wellesley email


## Additional Readings

Advanced Topics:
* [A Conceptual Introduction to Hamiltonian Monte Carlo](https://arxiv.org/pdf/1701.02434)


Ethics:
* [Predicted benefits, proven harms](https://thesociologicalreview.org/magazine/june-2023/artificial-intelligence/predicted-benefits-proven-harms/)


## Future Materials

* Geometry of Bayes' rule and law of total probability for continuous distributions
