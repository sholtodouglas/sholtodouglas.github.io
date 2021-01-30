---
layout: post
title: Transfer learning from play and language - nailing the baseline
categories: [play, language, imitation, latent]
author: Tristan Frizza
---

![alt-text-1](https://sholtodouglas.github.io/images/play/awesome.gif "demo of multiple tasks")


> [Code found here](https://colab.research.google.com/github/sholtodouglas/learning_from_play/blob/master/languageXplay.ipynb). 
> I worked hand in hand with [Tristan Frizza](https://twitter.com/TristanVtx) on this, and we had our questions both technical and directional patiently answered by Corey Lynch, Suraj Nair and Eric Jang. 

* TOC
{:toc}

# Introduction


> "Can we enable fast transfer learning to new scenes or behaviours by using language to structure a joint trajectory embedding space between robot specific data and much larger, diverse set of human video?"




### Introduction
We introduced the question we are trying to answer in [Laying down the infrastructure](https://sholtodouglas.github.io/LearningFromPlayAndLanguage/), but at that time we hadn't yet nailed baseline re-implementation of the paper [Learning from play (LFP)](https://learning-from-play.github.io/). We finally have, and wanted to write down a few of the key changes. 

Once again - the answer wasn't in neat regularisation techniques, interesting rotation representations or adding proprioceptive features, it lay in core fixes:
- Encouraging specific biases in the demonstration data
- Venturing beyond the plateau in our training curves
- Diagnosing overregularisation 
- Fixing gradient instabilities

We'd still like to explore more fun ideas (e.g, composing plans as a sequence of quantised latent vectors like VQ-VAE represents images as a sequence of quantised tiles - we think this may lead to a valuable decomposition of parts of skills, e.g sharing grasp encodings between objects or parts of the environment) - but for the moment, we've got our baseline and will move on to our original question!

### Learning to play again

Artist Michael Johnson once told me that it took him years to re-learn how to paint with the abstract creativity of a child - and it certainly took us months to learn how to 'play'. 

Take a look at this side by side comparison of the original paper's teleoperated 'play', and our initial dataset. While we did both perform a similar diversity of tasks, they interact with objects far more times in a row -  we typically performed one interaction then moved to the next. What this meant is that a bias was burned into the plans the model learned to immediately 'zoom away' following an attempted behaviour. Worse - if we weren't careful in teleoperating then there were patterns in how we moved (it is very tempting to push the button after the door). 

![alt-text-1](https://sholtodouglas.github.io/images/play/sidebyside.gif "side by side comparison")

This can be bandaged over shortening the re-plan interval - but our preference is for a model where the bias is 'fix up the object you just interacted with'.  Recollecting the data in this high-contact way dramatically improves how robust and accurate the model when interacting with the environment.

### What lies beyond the plateau?

![alt-text-1](https://sholtodouglas.github.io/images/play/graphs.png "demo of multiple tasks")
![alt-text-1](https://sholtodouglas.github.io/images/play/promised_land.png "demo of multiple tasks")
