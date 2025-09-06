
# Probabilistic Foundations of Machine Learning (CS345) 


## License 

Probabilistic Foundations of Machine Learning (c) 2024 by [Yaniv Yacoby](https://yanivyacoby.github.io/) is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/?ref=chooser-v1).


## Installation

1. Install `micromamba` with `brew`. Initialize the shell with: `micromamba shell init -s zsh`.

2. Create the environment:
   ```
   mm create -n book -c conda-forge python=3.10 -y
   pip install -r requirements.txt
   ```
3. Activate environment: `mm activate book` or run `pip install -U "jax[cpu]" numpyro chex scikit-learn pandas numpy matplotlib dill dominate pyyaml markdown Pillow jupyterlab jupyter-book sphinx-new-tab-link sphinx-proof tqdm ipdb IProgress -c https://tk.deepnote.com/constraints3.10.txt`

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
4. All figures are in [this directory](https://www.canva.com/folder/FAFIo00ejB4). 

## DeepNote Setup

1. Add student to workspace as editor
2. Ask student to duplicate homework project as private
3. Rename project "Homework N: FIRST-NAME LAST-NAME"
4. Click share, and invite instructor with edit access
5. To download a PDF version of your notebook, follow [these instructions](https://deepnote.com/docs/export-pdf)



## Docker

1. After updating the `requirements.txt` file, run `make docker` to recreate the docker image.
2. Then, tag the new image with a new version number: e.g., `docker tag yanivyacoby/wellesley-cs345 yanivyacoby/wellesley-cs345:v0`.
3. Finally run push the tagged version: e.g. `docker push yanivyacoby/wellesley-cs345:v0`.




## Additional Readings

Advanced Topics:
* [A Conceptual Introduction to Hamiltonian Monte Carlo](https://arxiv.org/pdf/1701.02434)


Ethics:
* [Predicted benefits, proven harms](https://thesociologicalreview.org/magazine/june-2023/artificial-intelligence/predicted-benefits-proven-harms/)


## Future Materials

* Geometry of Bayes' rule and law of total probability for continuous distributions
