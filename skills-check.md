# Skills Check

This course assumes knowledge of the topics and skills listed below. Although you may not have experience with every single one of these, we expect that by satisfying the prerequisites, you can self-learn whatever you're missing prior to the start of the course. **Please make time to go through all of the topics and tutorials below.**


## Coding

**Programming Language: Python.** The course is taught entirely in Python. To get the most out of the course, it's important you don't feel bogged down by too many syntax errors and the like so that you can focus on the concepts we introduce. If you do not have prior experience with Python, that's no problem. *Please do take the time before the start of the course to study and practice it.* For review, we recommend going through an online Python tutorial. For example, from this [interactive Python tutorial](https://www.learnpython.org/),
* Review all sections under "Learn the Basics" and "Data Science Tutorials."
* Review "List Comprehensions" and "Lambda functions" from the "Advanced Tutorials." 



**Libraries.** We will regularly use two Python libraries throughout the course: `pandas`, used for reading in `.csv` data files and manipulating them, and `matplotlib`, used for visualizations.
* To learn `pandas`, you can find lots of tutorials online. For example you may find the first two lessons from [this tutorial](https://www.kaggle.com/learn/pandas) helpful.
* Similarly, [this tutorial](https://www.w3schools.com/python/matplotlib_getting_started.asp) will walk you through how to use `matplotlib`.


## Coding Environment: DeepNote

We will be using [DeepNote](https://deepnote.com/) for all assignments in the course. DeepNote is an interactive Python environment that makes it easy to write code in small pieces, and to visualize and inspect them. Further, it lets you interleave your code with notes and math. If you're familiar with Jupyter notebooks, they are just a cloud version of a Jupyter notebook. If this is new to you, no problem! To get you started, we recommend:
* Making an account on the [DeepNote](https://deepnote.com/) website using your *Wellesley email* and exploring.
* Watching/reading some online tutorials, like [this one](https://www.youtube.com/watch?v=EW4lKlUnLGU).
* Finally try making cells that run code and ones for writing text and math. Follow [this guide](https://gtribello.github.io/mathNET/assets/notebook-writing.html) for formatting text and for writing mathematical notation---we will use this heavily in the course!
  * **Markdown** is language used for formatting is called "markdown." You can find a markdown cheatsheet [here](https://www.markdownguide.org/cheat-sheet/).
  * **Latex** (pronounced "la-tek") is the language used for typesetting mathematical notation. If you ever forget what's the syntax for a certain math symbol there are two ways for you to figure it quickly (aside from googling): (a) You can draw it in [this tool](https://detexify.kirelabs.org/classify.html) and it will tell you, or (b) you can right-click on any math on this website and select option "Show Math As"/"Copy to Clipboard", and then "TeX Command" to get the corresponding syntax. 



## Math

**Notation.** We will be using the notation for sums and products heavily throughout the course:
\begin{align*}
\sum\limits_{n=1}^N f(x_n) &= f(x_1) + f(x_2) + \dots + f(x_N) \\
\prod\limits_{n=1}^N f(x_n) &= f(x_1) \cdot f(x_2) \cdots f(x_N) 
\end{align*}

**Rules of Logarithms.** We will rely on the following:
\begin{align*}
\log (x \cdot y) &= \log x + \log y \\
\log x^y &= y \cdot \log x
\end{align*}

**Indicator Variables.** We will use indicator variables:
\begin{align*}
\mathbb{I}(\text{condition}) &= \begin{cases}
1 & \text{if the condition is true} \\
0 & \text{otherwise}
\end{cases}
\end{align*}
For example, $\mathbb{I}(A = a)$ will evaluate to 1 only when $A = a$. 

**Single Variable Calculus.** Derivatives, integrals, and limits (all in 1-dimension) will appear in some of the derivations. We will not ask you to do these by hand, but we expect you to know what they mean, conceptually and intuitively. To review these concepts, we recommend reading Chapters 3, 4, 8, 16, and 20 from [this tutorial](https://www.elevri.com/courses/calculus).



## Probability and Statistics

Although the course introduces a *probabilistic* paradigm of ML, *we do not assume any prior knowledge of probability and statistics*. All necessary topics from these subjects will be introduced as needed in the course.

