# How to Download/Run/Submit



## Joining the Class on Gradescope

1. Create an account on [Gradescope](https://www.gradescope.com/) using your *Wellesley College* email.
2. Join the course using the entry code `J776Z8`.



## Downloading the Homework

1. Log into [DeepNote](https://deepnote.com/) using your Wellesley College email. 
2. Go to homework link.
3. Duplicate the homework as *private*. You can do this by clicking on the "..." on the top right corner of the screen, and selected "Duplicate project". When the dialog box pops up, make sure to click the 'make private' button.
4. After the project has been duplicated. Rename it to say: `HW-N: FIRST LAST`, where `N` is the homework number, and `FIRST` and `LAST` are your first and last names.



## Submitting Homework

**Uploading.**
1. Run the notebook from start to finish (with no interruptions) so we can see the output of every cell.
2. Download the `.ipynb` file of your notebook from DeepNote.
3. Navigate to the course's Gradescope and submit it there.

**Checking.** Check to make sure that, in your *Gradescope* submission,
1. The `.ipynb` is rendered by Gradescope, and that it's not "too big" to be shown. If it's too big to be previewed, try:
  * Compressing the images you added as per the instructions below
  * Decreasing the number in `plt.rcParams['figure.dpi'] = 100` at the top of your notebook, e.g. to 50.
2. All cells have been run and their output is visible.
3. All images you included are showing.

Homework problems that do not include the output of the code, as described above, create substantial additional work for us to grade, and **will therefore be given zero credit**.



## Including Graphics in DeepNote

For some of the assignments, you'll have to draw figures (by hand, in google slides, or in some other way). To include these figures in a DeepNote notebook, follow these steps:
1. Compress your image by using [this website](https://imagecompressor.com/). 
2. Upload your compressed image to the `figs` directory in DeepNote.
3. Make sure this import statement is included: `from IPython.display import Image`.
4. In a *code* cell, write `Image('figs/FILENAME')`.



## Running Code Locally

The benefit of working in DeepNote is that no student should have installation issues---the environment is uniform for all students, and we've already taken care to ensure everything works. Students who prefer to complete the assignments locally on their own machines can follow the instructions here. However, we note that we unfortunately cannot help troubleshoot installation issues.

1. Create a virtual python environment with python $3.10$. We recommend using [Micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html) for this.
2. Then run the following commands in your terminal:
```
# Install Python packages
wget -O "requirements.txt" "https://raw.githubusercontent.com/mogu-lab/cs349-fall-2024/master/requirements.txt"
pip install -r ./requirements.txt

# Get the course's helper libraries
wget -O "deepnote_init.py" "https://raw.githubusercontent.com/mogu-lab/cs349-fall-2024/master/scripts/deepnote_init.py"
python deepnote_init.py

# Create a directory for uploading figures
mkdir -p figs
```

