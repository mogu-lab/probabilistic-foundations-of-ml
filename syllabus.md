# Syllabus


```{contents}
:local:
```


## Goals

Our goal is that at the end of this course, you will be able to:
* Formalize assumptions about an observed data-set into a statistical model.
* Perform statistical inference for statistical models in a probabilistic programming language.
* Evaluate models in the context of their downstream task by relating human-useful properties to metrics (and understanding their strengths/shortcomings).
* Abstractly reason/intuit about properties of models and different types of inference methods.
* Map complex, state-of-the-art models/inference methods back to simpler versions covered in class.
* Perform a broader impact analysis.
* Engage with Probabilistic ML research.



## What You Should Know About Us (the Course Staff)

We're here because we care about your experiences as students and as individuals. So what does that mean? It means that...

**We want you to come to our office hours.** We want to get to know you all, both in the context of the class and as individuals; we want to know what you find most challenging and what you find most rewarding in the class and in the CS program. Come chat with us!

**We want to see you _even more_ when you're struggling, lost, or down.** You may be surprised to know that as course staff, we still _vividly_ remember our struggle learning the material covered by this course. We recognize that there are "constructive" types of struggle---struggle that builds you up and helps you grow---as well as "destructive" types of struggle---struggle that tears you down, that leads you to question your self-worth, and that makes you want to give up. When things get hard, the worst thing you can do is isolate yourself, and the best thing you can do is to come talk to us. *Together, we can celebrate your growth and productively overcome the setbacks.*

**We have high expectations for you to be thoughtful and engaged.** We expect you to be thoughtful in your engagement with the course materials and in your interaction with your peers. Substantial amounts of research have shown that [Computer Science programs can create cultures of isolation and impostorism](https://courses.cs.washington.edu/courses/cse590e/02sp/defensive_20climate.pdf), in which students feel judged for common struggles many experience, instead of coming together to support one another. We ask you to be thoughtful in how you validate your peer's identities and create a safe and supportive space for everyone.

**We want you to find meaning in the course, and we want you succeed.** As in the liberal arts tradition, in addition to providing you with practical skills, we want this course to offer you *new lenses* with which to engage with the world; we want this course to be *personally meaningful* to you, to shape the way you think and engage with *societal issues* you see around you. 

**We designed this course *especially* for the Wellesley context.** We therefore designed the course based on several guiding principles.
* **Math, statistics, and computer science can all be intimidating, and they shouldn't be.** In fact, in recent years, [math anxiety](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6345718/) has gained attention as a phenomenon that needs to be understood and addressed in STEM education. It is important for us to challenge exclusionary definitions of what science is and who can practice it to create a supportive space for learning. None of these subjects require inborn abilities to succeed. We will therefore normalize the (very normal) struggles of learning this material in the class. 
* **Math, statistics, and computer science were invented by humans and are therefore shaped by human values.** We encourage you to question all course content, its history, its implicit values, its benefit to society, and more. We will create dedicated time and space for doing exactly this.
* **We tend to remember things we experience.** We believe people learn best by hands on practice, and by trial and error. We have therefore designed the class to be interactive. Further, we believe that knowledge is best learned with peers in societally meaningful contexts; we therefore incorporate lots of opportunities to engage with your peers in the class and to think about the techniques we introduce in a real-like context.
* **All math and theory we introduce will be practically useful.** We designed the course so it is always clear *why* math/theory is introduced, and how it is practically relevant. 
* **We want to know about your experience in the course.** We always strive to improve the course for future generations of students. We therefore value your feedback. 


## How to Succeed in CS 349?


**Come to class and participate.** While thorough, the course materials is *not a substitute* for the classroom experience. We expect you to attend all classroom sessions, engage with the lecture components and with your peers in the in-class exercises. 


**Attending office hours is an expected and normal part of the learning experience.** Moreover, office hours are an opportunity for you *take ownership of your own learning experience*---to ask questions you that will help you better understand the material, connect it to topics you personally care about, etc. We expect to see you at office hours regularly, asking questions, engaging with the materials, and supporting your peers. 


**Embrace confusion.** Confusion may be uncomfortable; it may cause us to doubt whether we have the skills necessary to make it through the course. It is therefore our mission to help you find ways to embrace confusion *productively*, because without it, there's no learning. 


**Experiment and tinker.** Learning requires that you form your own mental model of the material, and your own intuition for how things work. One effective way to do this is through play---if you're not sure what would happen if you tweak a piece of code, a parameter of an ML model, etc. try it! 





**Ready to take CS 349?** Fasten your seatbelts---*it's going to be an adventure!*

```{image} img/banner.png
:alt: Cats eating popcorn
:width: 500px
:align: center
```




## Classroom Environment

**Diversity, Equity, and Inclusion (DEI):** It is the mission of the teaching staff that students from all diverse backgrounds and perspectives be well served by this course, that students' learning needs be addressed both in and out of class, and that the diversity that students bring to this class be viewed as a resource, strength, and benefit. We aim to create a learning environment that is inclusive and respectful of diversity: gender, sexuality, disability, age, socioeconomic status, ethnicity, race, and culture. Your suggestions for how to better our classroom community are always encouraged and appreciated.

Since a large part of this course requires students to work in groups, in alignment with our teaching mission, we ask that students explicitly reflect on and implement practices for building teams that are diverse along many axes. The teaching staff is happy to help you brainstorm how to create an inclusive and productive working culture for your team.


**Mental Health:** We value your mental health more than anything else. If you're finding yourself facing mental health challenges, please come talk to us. As instructors, we have had our own fair share of mental health challenges. We are happy to help you find the support you need, whether on- or off-campus---our door is always open!





## Grades

Course grades are computed by weighing course components as follows:
* Assignment work: $45\%$
* Final project: $40\%$
* In-class participation and exercises: $15\%$

Course grade percentages are converted to letters via the table below:

| **Percentage** | **Letter Grade** |
|---|---|
| $\geq 95.0\%$ | A |
| $\geq 90.0\%$ | A- |
| $\geq 86.6\%$ | B+ |
| $\geq 83.3\%$ | B |
| $\geq 80.0\%$ | B- |
| $\geq 76.6\%$ | C+ |
| $\geq 73.3\%$ | C |
| $\geq 70.0\%$ | C- |
| $\geq 60.0\%$ | D |
| $< 60.0\%$ | F |


In case the average grade in the class is lower than expected, we may raise everyone's grades. We will *never* lower the grades.

Recall that the minimum grade to earn credit under credit/no credit grading is C. The minimum grade to pass and earn credit under normal letter grading is D.




## Late Policy

TODO



## Honor Code and Collaboration/Resource Policies

TODO



## Accessibility and Disability

Every student has a right to full access in this course. If you need any accommodations for CS 349, please contact Wellesley’s Disability Services. You should request accommodations as early as possible during the course, since some situations can require significant time for accommodation design. If you need immediate accommodations, please arrange an appointment with me as soon as possible. If you are unsure but suspect you may have an undocumented need for accommodations, you are encouraged to contact Disability Services. They can provide assistance including screening and referral for assessments. Disability Services can be reached at `disabilityservices@wellesley.edu`, at `781-283-2434`, or by [scheduling an appointment online](https://www.wellesley.edu/disability).


## Religious Observance

Students whose religious observances conflict with scheduled course events should contact the instructors in advance to discuss alternative arrangements. You may do this through the [Wellesley College Religious Observance Notification System](https://webapps.wellesley.edu/religious_calendar/) if you prefer.



## Policies on Discrimination, Harassment, and Sexual Misconduct

Wellesley College considers diversity essential to educational excellence, and we are committed to being a community in which each member thrives. The College does not allow discrimination or harassment based on race, color, sex, gender identity or expression, sexual orientation, ethnic or national origin or ancestry, physical or mental disability, pregnancy or any other protected status under applicable local, state or federal law.

If you or someone you know has experienced discrimination or harassment, support is available to you:
* **Confidential Reporting:** Students can report their experiences to [Health Services](https://www.wellesley.edu/healthservice) (`781-283-2810`); [Stone Center Counseling Service](https://www.wellesley.edu/counseling) (`781-283-2839`); or [Religious and Spiritual Life](https://www.wellesley.edu/campuslife/community/religiousandspirituallife) (`781-283-2685`). *These offices are not required to report allegations of sexual misconduct to the College.*
* **Non-Confidential Reporting:** You can let either of your CS 349 instructors know. As faculty members, we are obligated to report allegations of sex-based discrimination to the [Non-discrimination/Title IX Office](https://www.wellesley.edu/administration/offices/titleix).
  * You can report directly to the [Non-discrimination/Title IX Office](https://www.wellesley.edu/administration/offices/titleix) (`781-283-2451`) to receive support, and to learn more about your options for a response by the College or about reporting to a different institution.
  * You can report to the [Wellesley College Police Department](https://www.wellesley.edu/police) (Emergency: `781-283-5555`, Non-emergency: `781-283-2121`) if you believe a crime has been committed, or if there is an immediate safety risk.




## Acknowledgements

This syllabus draws on [CS 240's syllabus](https://cs.wellesley.edu/~cs240/s24/about/), and Evan Peck's [What I want you to know about me as your professor](https://medium.com/bucknell-hci/what-i-want-you-to-know-about-me-as-your-professor-58c9c2e91e33).

