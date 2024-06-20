
# Probabilistic Foundations of Machine Learning (CS349), Fall 2024

## Installation

```
conda create --name book python=3.10 -y
pip install jupyter-book 
pip install ipython ipykernel pandas numpy matplotlib scipy seaborn scikit-learn
pip install ipdb jupyterlab tqdm ipywidgets
```


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

