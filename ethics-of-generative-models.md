# The Ethics of Generative Models in Sociotechnical Systems


**Contexts:** We can evaluate predictive models in terms of their prediction error, confusion matrices, etc. Moreover, we can begin to assess for discrimination by comparing these notions of predictive error across different subgroups in our data. Can we do the same for generative models?

**Challenge:** Generative models pose new challenges for evaluation. Log-likelihood allows us to compare different generative models, but how can we tell if a model's log-likelihood is sufficiently high for some downstream task? And how can we determine whether the generative model learned generates biased data (e.g. data that reinforces stereotypes)? 

**Outline:** 
* We'll start with a broader impact analysis of some case studies
* We'll look at some interdisciplinary approaches to assessing generative models



## Broader Impact Analysis

````{admonition} Exercise: Conduct a Broader Impact Analysis
In this exercise, we'll be looking at the ways in which Generative AI models reproduce and reinforce systemic racism, sexism, etc. These readings show that, the biases exhibited by these models are *even worse* than the biases we see in real life. For example, the portion of doctors identifying as women is already significantly smaller than the portion of doctors identifying as men. However, when asked to generate portraits of doctors, AI will generate portraits in which gender proportions are even more skewed. 

Read **one** of the following:
* [Stable Bias: Evaluating Societal Representations in Diffusion Models](https://arxiv.org/pdf/2303.11408).
* [Humans are Biased; Generative AI is Even Worse. Stable Diffusion’s text-to-image model amplifies stereotypes about race and gender---here's why that matters](https://www.bloomberg.com/graphics/2023-generative-ai-bias/). **Content Warning:** In contrast to the other reading, this reading presents the topic using AI-generated visualizations of racist, sexist stereotypes that may be viscerally triggering.

Then answer these questions:
* Conduct a broader impact analysis of the technology described (text-to-image stable diffusion models).
* What did you find challenging about conducting this broader impact analysis?
````


## The "Price" of Generative AI

````{admonition} Exercise: The "Price" of Generative AI
In this exercise, we will extend the broader impact analysis you conducted in the problem above to better understand some of the "hidden costs" associated with Generative AI.
* Read: ["It’s destroyed me completely": Kenyan moderators decry toll of training of AI models](https://www.theguardian.com/technology/2023/aug/02/ai-chatbot-training-human-toll-content-moderator-meta-openai). **Content Warning:** This article discusses experiences of psychological trauma among content moderators, including references to graphic, violent, and sexually explicit material.
* Watch: [I Live 400 Yards From Mark Zuckerberg’s Massive Data Center](https://www.youtube.com/watch?v=DGjj7wDYaiI).

Then answer:
* Was there anything that surprised you about the reading/video?
* Go back to your broader impact analysis above and revise it based on the reading and video.
````


