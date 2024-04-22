# Probabilistic Foundations of ML

```{warning}
This course is currently under construction!
```

**Instructor:** [Yaniv Yacoby](https://yanivyacoby.github.io/) (he/him)

**Semester:** Fall 2024

**Course Number:** CS 349 @ Wellesley College

**Description:** In recent years, Artificial Intelligence has enabled applications that were previously not thought possible—from systems that propose novel drugs or generate new art/music, to systems that accurately and reliably predict outcomes of medical interventions in real-time. But what has enabled these developments? Probabilistic Machine Learning, a paradigm that casts recent advances in Machine Learning, like neural networks, into a statistical learning framework. In this course, we introduce the foundational concepts behind this paradigm—statistical model specification, and statistical learning and inference—focusing on connecting theory with real-world applications and hands-on practice. This course lays the foundation for advanced study and research in Machine Learning. Topics include: directed graphical models, deep Bayesian regression/classification, generative models (latent variable models) for clustering, dimensionality reduction, and time-series forecasting. Students will get hands-on experience building models for specific tasks, most taken from healthcare contexts, using a probabilistic programming language based in Python.


```{contents}
:local:
```



## Goals



## Topics



## Frequently Asked Questions


**Is previous experience with Machine Learning required for this class?**

> Nope! The class is designed for students *without* a Machine Learning background.


**How will the class rely on the listed math prerequisites?**

> The main purpose of the math prerequisites for CS 349 is to ensure that students have had enough exposure to some math so that the new content we introduce is not overwhelming. We will not assume much specific knowledge is needed from the listed prerequisite classes.


**Can I audit the class?**

> Unfortunately, given the amount of interest in the class, we cannot allow you to audit.


**What kind of healthcare context will be used in the class?**

>Since acquiring access to real healthcare data-sets in the context of a class is tricky, we will instead use data from the "Intergalactic Hypothetical Hospital," a hospital we made up especially for this class, where we will be analyzing data to improve medical care for intergalactic beings.

**Is this class better for students wishing to go to industry or grad school?**

> The course will set students up to be able to develop and interrogate probabilistic ML models, which can be helpful in both contexts.


**How difficult will this class be?**
> Since this is a new course, developed based on a new philosophy for teaching probabilistic ML, it is difficult to say! We recommend not assuming it will be easy :).


**What's the overlap with CS 305 and MIT 6.390?**

> There are several main differences between CS 349, and CS 305 and MIT 6.390. The first is that CS 305 and MIT 6.390 both take an _optimization perspective_ on ML; that is, we propose an objective/loss function that, when minimized, will give us a model that (hopefully) fits the data. In contrast, CS 349 takes a _probabilistic perspective_, which uses statistics to specify the model, and uses principles of frequentist/Bayesian learning to fit the model to data. Thus, even though there's a small number of topics that overlap, these topics will be presented from two different perspectives. Second, CS 305 and MIT 6.390 focus more on the inner workings of deep learning architectures (e.g. how convolutional vs. recurrent neural networks work), whereas in CS3 349 treats deep learning architectures abstractly as an expressive modeling tool. In CS 349, we will then use deep learning as building blocks, and work our way towards deep Bayesian regression/classification---model that can tell us how "unsure" they are about their predictions (e.g. Bayesian Neural Networks)---and deep generative models (e.g. Variational Autoencoders, Diffusion Models).


**What does a typical day look like in CS 349? What will the workload look like?**

> A typical class will dedicate half the time to the lecture, and the other half to in-class activities that students will bring home to finish as homework. There will be regular homework, in-class activities, and a final project. At the moment, we're not planning on having exams.


**What language/coding environment will we use?**

> The class will be taught in Python. We will be using Google Collab or Deepnote for assignments – these are coding "notebooks" that allow you to write code, make visualizations, take notes, and write math all in the same place (they are really interactive and fun to use!). There are also two Python libraries we will especially rely on. The first is [Jax](https://jax.readthedocs.io/), which enables very fast array operations (which is the basis for all ML), and the second is [NumPyro](https://num.pyro.ai/), which will allow us to translate the math/stats that describe our ML model into code we can use to fit to data.


**What's the split between math and coding?**

> On the math side, the class will introduce the necessary math and stats (from probability, linear algebra, and calculus) needed to create fairly complicated probabilistic ML models – whether this will feel math-heavy is a little hard to tell a priori. On the coding side, unlike other coding-heavy classes, CS 349 will not require writing a lot of code (in terms of the number of lines), but the code we will write will be conceptually tricky and will require new skill sets to debug. In that sense, it's a coding-heavy class.


**How often will CS 349 be offered?**

> The current plan is: once a year! (though the course number and title might change as we figure out how it best fits into the broader Wellesley Computer Science and Data Science curriculums).
  



![banner](img/banner.png)

