---
layout: post
title: Transfer Learning from Play and Language - Nailing the Baseline
categories: [play, language, imitation, latent]
author: Tristan Frizza
---

![alt-text-1](https://sholtodouglas.github.io/images/play/awesome.gif "demo of multiple tasks")


> [Code found here](https://github.com/sholtodouglas/learning_from_play). 
> I worked hand in hand with [Tristan Frizza](https://twitter.com/TristanVtx) on this, and we had our questions both technical and directional patiently answered by Corey Lynch, Suraj Nair and Eric Jang. 

* TOC
{:toc}

# Introduction

This is part of an ongoing series where Tristan and I are trying to re-implement the [Learning from play (LFP)](https://learning-from-play.github.io/) line of research, then build on it to answer a couple of questions, above all:

> "Can we enable fast transfer learning to new scenes or behaviours by using language to structure a joint trajectory embedding space between robot specific data and much larger, diverse set of human video?"

We've finally nailed a great baseline re-implementation. In the gif above you can see goals specified by the transparent copies of the object - and it is capable of reliably completing > 10 different tasks in a row. You can read a little bit more about their papers, the question we are trying to answer and getting the environment right in [Laying down the infrastructure](https://sholtodouglas.github.io/LearningFromPlayAndLanguage/).

### Heres what it took

Once again - the answer wasn't in neat regularisation techniques, interesting rotation representations, proprioceptive features, learnt initialisations for the LSTM or any of the other highly specific things we tried in response to particular deficiencies -  it lay in core fixes:
- Realising how our 'play' was biasing the dataset
- Venturing beyond the plateau in our training curves
- Diagnosing overregularisation 
- Fixing gradient instabilities



### Learning to play again

I once heard that it takes abstract artists years to re-learn how to paint with the freedom and  creativity of a child - it certainly took us months to learn how to 'play'. 

Take a look at this side by side comparison of the original paper's teleoperated 'play', and our initial dataset. While we did both perform a similar diversity of tasks, they interact with objects far more times in a row -  we typically performed one interaction then moved to the next. What this meant is that a bias was burned in to immediately 'zoom away' following an attempted behaviour. Worse - if we weren't careful in teleoperating then there were patterns in how we moved (it is very tempting to push the button after the door). 

![alt-text-1](https://sholtodouglas.github.io/images/play/sidebyside.gif "side by side comparison")

This can be bandaged over by shortening the re-plan interval - but our preference is for a model where the bias is 'fix up the object you just interacted with'.  Recollecting the data in this multi-interaction way dramatically improves how robust and accurate the model is. The 'post interaction' phase of a plan initialises the next plan with an ideal starting point for retrying (on failure), or fixing up (on partial success). 

Interestingly it is only due to the behaviour itself. We suspected a multi-interaction dataset may also provide more timesteps of interaction with the environment and so counted the proportion of timesteps where an environment variable was different to the previous state (i.e, arm interacting not transitioning), but the difference was neglible. 

### What lies beyond the plateau?

This one is a little obvious in retrospect. Train longer! We used Colab TPUs for all of our training, and it just so happens that the point at which we break away from the plateau is just after the typical timeout. As we were doing all our experiments in series due to compute restrictions - it always felt more important to try another experiment. This is compounded by the fact that there is a relatively narrow range of Beta values (the relative weighting between the regularisation term and the action reconstruction term) which work. Too high, and the regularisation loss never increases and the latent space collapses. Too low, and it would take even longer than it did for the regularisation loss to bend down and allow the planned trajectories to match up to the encoded ones. We found B=0.00003 worked well. In the paper, they use B=0.01. This is because they use log-liklihood loss on the actions, wheras we use MAE (with a commensurately lower magnitude). 

![alt-text-1](https://sholtodouglas.github.io/images/play/conver.gif "demo of multiple tasks")

### Diagnosing Overregularisation

### Fixing Gradient Instabilities

### Whats next?


We'd still like to explore more fun ideas (e.g, composing plans as a sequence of quantised latent vectors like VQ-VAE represents images as a sequence of quantised tiles - we think this may lead to a valuable decomposition of parts of skills, e.g sharing grasp encodings between objects or parts of the environment) - but for the moment, we've got our baseline and will move on to our original question!


![alt-text-1](https://sholtodouglas.github.io/images/play/behaviour_space.png "Quelle separation")
