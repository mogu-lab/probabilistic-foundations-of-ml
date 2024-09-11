# The Ethics of Machine Learning: A View from History

**Context:** In the previous chapters on ethics, we've considered how societal values shape data collection and modeling choices. We will now dive more deeply into the historical context from which many ideas behind statistics and probabilistic ML emerged. We will use this historical context as a way to reflect and question the ethics behind our own practice. 

**Challenge:** The specific movement we will examine is eugenics. We recognize that many of the topics here may be uncomfortable to you---we will read documents concerned with offensive discriminatory ideologies. We have chosen these topics because we consider them important to modern discussions of ethics in ML/AI. We ask that you maintain open communication with us about how you're feeling as you make your way through these readings and exercises. 

**Outline:** 
* What was the eugenics movement?
* The birth of modern statistics
* The politics of ML
* Science is a human endeavor
* What can we learn from this?


## What is the Eugenics Movement?

````{admonition} Exercise: Background on Eugenics
**Part 1:** Read, *[Eugenics and Scientific Racism](https://www.genome.gov/about-genomics/fact-sheets/Eugenics-and-Scientific-Racism)*.
* What is the goal of eugenics / scientific racism?
* What form did eugenics take in the US? In Nazi Germany?
* Who participated in the eugenics movement?

**Part 2:** From, *[The Mismeasure of Man](http://tankona.free.fr/gould1981.pdf)*, read "The allure of numbers" and "Masters of craniometry: Paul Broca and his school" from Chapter 3. **Content Warning:** This text provides a criticism of eugenics. It does so using *dated* language and by explicitly referencing elitist, white supremacist ideas and visualizations. 

Then, answer:
* What's the "allure of numbers"?
* What does the author mean by "Science is rooted in creative interpretation"? Provide an example (of the "numbers," interpretation, and theory). 
* What were some of the things Francis Galton quantified? What are some fallacies of conclusions he might draw from his data?
* What's Craniometry? 
* What happened when Mall tried to replicate Bean's findings? 
* What were the fallacies of Broca's scientific methods?

**Part 3:** Reflection.
* Did you find anything surprising about these readings?
* How did these readings make you feel?
* What did they make you reflect on?
* What questions do you have?
````



## The Birth of Modern Statistics


When reading about Craniometry in *The Mismeasure of Man*, it's hard not to wonder what was going on through those scientists' minds. It feels like their science is more rooted in desperation than it is in the real world--- desperation to justify to others their superiority as white, European, cis-male academics. And to do so in a way that is most "objective" by rooting their arguments in numbers and statistics. It's also hard not to mock them for their ridiculous obsession with measuring and quantifying everything around them in hopes of arriving at some universal formulae. Finally, in viewing eugenics in this way, we may be tempted to dismiss eugenics as a "blip" in history---a period marked by un-scientific practices used to justify clearly incorrect beliefs (e.g. that intelligence can be measured from brain sizes, that there exists some ranking of humans). 

However, here, we argue that dismissing eugenics as bad science is irresponsible. As we will show next, eugenics has fundamentally shaped statistics, science, and policy for years to come.


````{admonition} Exercise: The Development of Modern Statistics
**Part 1:** Read, *[How Eugenics Shaped Statistics](https://nautil.us/how-eugenics-shaped-statistics-238014/)*. **Content Warning:** As the name suggests, this article describes how eugenics and statistics co-shaped one another. It explicitly references white supremacist language. 
* In what ways does the author claim eugenics shaped statistics?
* Who were Galton, Pearson, and Fisher? 
* Which statistical tools did each develop? And what purpose was each developed for?
* How did these scientists argue for the objectivity of their findings?
* What's the legacy of eugenics that's mentioned in the article?
* In what ways have scientists criticized the scientific legitimacy of eugenics since?

**Part 2:** Reflection.
* If you were alive during the height of the eugenics movement, do you think you would have bought into it? Why or why not?
* Did you find anything surprising about the reading?
* How did the reading make you feel?
* What did it make you reflect on?
* What questions do you have?
````


## The Politics of ML

Many of the techniques introduced in this class were created by leaders of the eugenics movement. To name a few, the MLE was proposed by Fisher in *[On an Absolute Criterion for Fitting Frequency Curves](https://www.jstor.org/stable/2246266)*. Factor Analysis was proposed by Burt in *[Experimental Tests of General Intelligence](https://bpspsychub.onlinelibrary.wiley.com/doi/10.1111/j.2044-8295.1909.tb00197.x)*, and [refined by Spearman](https://www.semanticscholar.org/paper/Charles-Spearman%2C-Cyril-Burt%2C-and-the-origins-of-Lovie-Lovie/b4f2fe99f093d7dce355b3a1aff15889b0e364c3). Linear regression was proposed by Fisher in *[The Goodness of Fit of Regression Formulae, and the Distribution of Regression Coefficients](https://www.jstor.org/stable/2341124)*. Correlation was proposed by Galton in *[Co-Relations and Their Measurement, Chiefly from Anthropometric Data](https://www.jstor.org/stable/114860)*. Given this, we have to ask ourselves: are there remnants of eugenics idealogy in our ML toolkit? 


````{admonition} Exercise: The Politics of Artifacts
**Part 1:** Language, idealogy, and practice. 
* Many eugenicists advocated to "let the data speak for itself." Is this different from the language used to tout the benefits of Deep Learning, that it's "data-driven"? Why or why not? What are the potential harms of this idea?
* Many eugenicists advocated for the use of correlation analysis to find variables that have statistically significant relationships in the data. Does this practice lend itself more easily to confirmatory science (science used to justify what we already believe)? Why or why not? 
* Have you noticed other potential similarities between practices of eugenics and practices of data science / ML?

**Part 2:** Read, *[Do artifacts have politics?](https://faculty.cc.gatech.edu/~beki/cs4001/Winner.pdf)*
* How will the legacy of Robert Moses continue to shape New York City? Making a parallel to eugenics, how might the legacy of Galton, Pearson, and Fisher may continue to shape science? ML? 
* How does the author define "inherently political technologies"? Explain an example they provide. 
* How would the author argue that the autocomplete system embedded in our email is inherently political? 
* What argument can you make that statistical methods are politically neutral?
* What argument can you make that statistical methods aren't politically neutral?

**Part 3:** The politics of the math behind statistics and ML. 
* As the author of *[How Eugenics Shaped Statistics](https://nautil.us/how-eugenics-shaped-statistics-238014/)* argues, statistical tests have politics for two main reasons. The first is that, at its heart, it enforces a binarization: either something is or isn't statistically significant. What are the political implications of this? What are the potential harms of this?
* Second, statistical tests have politics because they encourage a fixation on difference, as opposed to magnitude. That is, you can find two quantities that are statistically significantly different, but in practice the fact they are different may be meaningless. What are the political implications of this? What are the potential harms of this?
* Do we find eugenics ideals embedded in the math of other methods? Let's consider the MLE. For a regression model, we find that the MLE gives us a model that minimizes the squared error between true and observed (see the section on non-probabilistic regression):
    \begin{align}
    \frac{1}{N} \sum\limits_{n=1}^N \left(y_n^\text{observed} - y_n^\text{predicted} \right)^2.
    \end{align}
    If our data set includes points from two populations, a majority and a minority population, which population will have a better-fitting model? What are the political implications of this? What are the potential harms of this?
* In your opinion, should we stop using these tools? Why or why not?

**Part 4:** Reflection.
* Did you find anything surprising about these readings/exercises?
* How did they make you feel?
* What did they make you reflect on?
* What questions do you have?
````



## Science is a Human Endeavor


It's important for us not just to study the ongoing harms of the movement, but to consider whether we, as scientists, are fundamentally different from the leaders of eugenics. Are we subject to the same types of biases and unscientific methods? 


````{admonition} (OPTIONAL) Exercise: Eugenics in AI Research
Read, *[The TESCREAL bundle: Eugenics and the promise of utopia through artificial general intelligence](https://firstmonday.org/ojs/index.php/fm/article/view/13636/11606)*.
* What's Artificial General Intelligence (AGI)? And what are the criticisms the author presents against building an AGI?
* What is the TESCREAL bundle?
* What is "second-wave eugenics"? How does the author connect the TESCREAL bundle to second-wave eugenics?
````



## What can we learn from this?

By this point, you may be thinking, why did we learn all of this ML just to tear this knowledge down in the end? Given the deep ties between ML and eugenics, how can we ethically and responsibly move forward as technologists? There are a few things we hope you took away from the readings and exercises so far. 

**Everything is political.** As we've seen in earlier units on ethics, data, modeling, and interpretation of results are not neutral. The questions we ask and the scientific process we follow are informed by our societal values, and are therefore not neutral. In the context of eugenics, the needs of the movement (e.g. statistics to justify differences between races), motivated the development of statistical methodology. Without these needs, perhaps Galton, Pearson and Fisher would have developed totally different tools? The methodology they developed itself---that is, the math---may implicitly enforce eugenics ideals (e.g. averages representing the "ideal" human). Everything can and should be interrogated from a variety of lenses. 

**More specifically, math and science are human-made.** All knowledge you learned in math, statistics, and computer science courses was created by humans. For example, the MLE does not exist in nature; it is not the universally "right" way to fit a probabilistic model (also human-made) to data. It was invented by a person a little over 100 years ago. It therefore deserves to be questioned and interrogated. 

**We want to empower you to ask the big questions.** We're not asking you to toss out everything you know; we're asking you to think deeply about the implicit assumptions behind the tools we use---to analyze them mathematically, unfold their consequences in sociotechnical systems, consider them in their historical context, etc., as we did with eugenics. Moreover, we argue that this is the *true spirit* of probabilistic ML---to use a variety of lenses to make our assumptions explicit so we can make informed, ethical, and responsible choices. 

**You don't need to have answers to these questions.** The questions we are raising---the questions we're asking *you* to raise---may not have good answers yet. These are open research questions that aren't for us to answer individually; they will require us to engage with other people with different perspectives, identities, and lived experiences. We ask you to embrace confusion and tension; we know it's hard, and we believe it's the way to productively move forward. 


````{admonition} Exercise: Reflection
* What did you learn from reading about the eugenics movement? 
* How will what you learned inform your approach to science?
````


<br/>

**Acknowledgements.** This chapter draws on [FASPE 2024](https://www.faspe-ethics.org/). 

