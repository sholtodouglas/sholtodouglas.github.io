---
layout: post
title: Transfer Learning from Play and Language - Nailing the Baseline
categories: [play, language, imitation]
author: Tristan Frizza
---

![alt-text-1](https://sholtodouglas.github.io/images/play/with_latent5.gif "demo of multiple tasks")

{% include image.html url="/images/play/with_latent5.gif" description="My cat, Robert Downey Jr." %}

> [Code found here](https://github.com/sholtodouglas/learning_from_play). 
> I worked hand in hand with [Tristan Frizza](https://twitter.com/TristanVtx) on this.

* TOC
{:toc}

# Introduction

This is part of an ongoing series where Tristan and I are trying to re-implement the [Learning from play (LFP)](https://learning-from-play.github.io/) line of research, then build on it to answer a couple of questions, above all:

> "Can we enable fast transfer learning to new scenes or behaviours by using language to structure a joint trajectory embedding space between robot specific data and a much larger, diverse set of human video?"

We've finally nailed a great baseline re-implementation. In the gif above you can see goals specified by the transparent copies of the object - it is capable of reliably completing > 10 different tasks in a row. You can read a little bit more about their papers, the question we are trying to answer and getting the environment right in [Laying down the infrastructure](https://sholtodouglas.github.io/LearningFromPlayAndLanguage/).

### Heres what it took

Once again - the answer wasn't in neat regularisation techniques, interesting rotation representations, proprioceptive features, learnt initialisations for the LSTM or any of the other highly specific things we tried in response to particular deficiencies -  it lay in core fixes:
- Realising how our 'play' was biasing the dataset
- Venturing beyond the plateau in our training curves
- Diagnosing overregularisation 
- Fixing gradient instabilities 



### Learning to play again

I once heard that it takes abstract artists years to re-learn how to paint with the freedom and  creativity of a child - it certainly took us months to learn how to 'play'. 

Take a look at this side by side comparison of the original paper's teleoperated 'play', and our initial dataset. While we did both perform a similar diversity of tasks, they interact with objects far more times in a row. We typically performed one interaction then moved to the next. What this meant is that a bias was burned in to immediately transition to another object following an attempted behaviour. Worse - if we weren't careful in teleoperating then there were regular patterns in how we moved (it is very tempting to push the button after the door). 

![alt-text-1](https://sholtodouglas.github.io/images/play/cut.gif "side by side comparison")

This can be bandaged over by shortening the re-plan interval - but our preference is for a model where the bias is 'fix up the object you just interacted with'.  Recollecting the data in this multi-interaction way dramatically improves how robust and accurate the model is. The 'post interaction' phase of a plan initialises the next plan with an ideal starting point for retrying (on failure), or fixing up (on partial success). 

To verify that this effect was due to the the behaviour demonstrated, and not that a multi-interaction dataset provides more timesteps of interaction with the environment - we counted the proportion of timesteps where an environment variable was different to the previous state (i.e, arm interacting not transitioning), but the difference was neglible.

### What lies beyond the plateau?

This one is a little obvious in retrospect. Train longer! We used Colab TPUs for all of our training, and it just so happens that the point at which we break away from the plateau is just after the typical timeout. As we were doing all our experiments in series due to compute restrictions - it always felt more important to try another experiment. This is compounded by the fact that there is a relatively narrow range of Beta values (the relative weighting between the regularisation term and the action reconstruction term) which work. Too high, and the regularisation loss never increases and the latent space collapses. Too low, and it would take even longer than it did for the regularisation loss to bend down and allow the planned trajectories to match up to the encoded ones. 

![alt-text-1](https://sholtodouglas.github.io/images/play/convergence.gif "demo of multiple tasks")

### Diagnosing Overregularisation [BETTER PLOTS TO COME]
Recall that there are two potential 'plan' inputs to the actor. 
- The output of the encoder over the full trajectory to reconstruct, a specific path from A-B
- The output of the planner when given only the current state and the goal state, from which you sample one potential path from A-B. 


Planner based action reconstruction loss is not a perfect indicator of end performance because if you overregularise it will be better early (as it is easier for the planner to match the encoder outputs) - but the latent space will be less informative and the ultimate performance will be limited. The encoder reconstruction loss (the lower plots of B0.0001 and B0.00003) is the lower bound of reconstruction loss. 
![alt-text-1](https://sholtodouglas.github.io/images/play/sweep.png "Regularisation Demonstration")

![alt-text-1](https://sholtodouglas.github.io/images/play/all_plots.png "Regularisation Demonstration")
As you can see, with a beta value which is too high the planner and encoder latent spaces begin converging too early, and will ultimately converge to a higher loss than that of a lower beta value - even though the planner action reconstruction loss is initially lower. Furthermore, goal-conditioned behavioural cloning (GCBC) - without any latent space to model the different ways of going from A-B has a great reconstruction loss! Despite this, it performs much worse on the actual environment. We believe this is because its possible to get to great reconstruction loss by outputting the mean of behaviour (in the same way that VAEs used to output blurry pictures), but to output one of the specific ways runs the risk of being very wrong (on the specific training example shown). As far as ultimate performance goes though - it doesn't really matter if the planner picks the wrong plan to follow (and thus has a commensurately higher loss), so long as all plans which it can pick from are effective (demonstrated by near perfect encoder reconstruction, with a reasonably and well aligned planner latent space). 

One of the easiest ways to diagnose this overregularisation is not actually the loss curves, but labelled plots of the latent space of behaviours (thanks for the tip Corey!). The latent space should become distinct quite early - and stay that way. Below is a plot of the latent space of plans generated by the planner on a sample of the test set. The distribution clearly aligns to the environment - take another look at the gif up top!
![alt-text-1](https://sholtodouglas.github.io/images/play/behaviour_space.png "Quelle separation")

### Fixing Gradient Instabilities [PLOTS TO COME]

It took us way too long to plot per-model gradient norm breakdowns. These revealed spikes in the gradient norms which disrupted the training curve (especially for probabilistic models), and we found that clipping these dynamically if it was > 4x the previous norm worked well.  

### Whats next?

First up, we've now set up TFRC and will do proper ablations - our previous comparisons all ended too soon. 

We'd still like to explore more fun ideas (e.g, composing plans as a sequence of quantised latent vectors like VQ-VAE represents images as a sequence of quantised tiles - we think this may lead to a valuable decomposition of parts of skills, e.g sharing grasp encodings between objects or parts of the environment) - but for the moment, we've got our baseline and will move on to our original question!

> Thank you to Corey Lynch, Suraj Nair and Eric Jang for patiently answering our questions.
