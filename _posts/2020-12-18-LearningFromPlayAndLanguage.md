---
layout: post
title: Learning from play and language [Work in Progress]
categories: [play, language, imitation, latent]
---

![alt-text-1](https://sholtodouglas.github.io/images/play/ai_9_1.gif "title-1")


> [Code found here](https://colab.research.google.com/github/sholtodouglas/learning_from_play/blob/master/languageXplay.ipynb). 
> ALl of this was done with [Tristan Frizza](https://twitter.com/TristanVtx). 

* TOC
{:toc}

# Introduction

Tristan and I were inspired to ask the question

> "Can we enable fast transfer learning to new scenes or behaviours by using language to exand the available training data for imitation learning?"

We were inspired to do this by a pair of papers, [Learning from play (LFP)](https://learning-from-play.github.io/) and [Grounding language in play (LangLMP)](https://arxiv.org/abs/2005.07648). These present a couple of important ideas:
- In a long dataset of teleoperated 'play', you can sample any subsequence as training data for a goal conditioned policy which learns to go from arbitrary A to B. This is vastly more data efficient and effective than any other form of imitation learning we've seen, a few hours of teleoperated 'play' provides millions of overlapping subsequences. 
- These trajectories are unlikely to be optimal, they are only one of the many ways of getting from A-B. This multimodality is captured by using a seq-seq VAE as the actor. The encoder of the VAE encodes the subsequence as a latent 'trajectory' vector $E(z \| a_0,s_0 .. a_T, s_T) $, which the decoder combines with the state and goal state at each timestep to choose an action $\pi(a \|z,s_t, s_g)$. During training, they seek to maximise the log probability of the true actions from that sequence. At test time, a 'planner' samples a potential trajectory vector given the initial and end goal $P(z \|s_0, s_g)$. During training, the actor tries to decode that specific way of achieving the goal - at test time, the actor decodes one of the many ways of achieving it. Use images rather than states as goals by embedding the image in a lower dimensional space with a CNN as a feature encoder.
- The model can be adapted to use sentences as a goal by labelling a small (<1% of total dataset in their experiments) number of trajectories. Goal images and sentences will share the same goal embedding space because they will correspond to the same sequences of actions, which will be maximised for by the same imitation learning objective as before. This allows language to control the robot, while still learning more robust control from a vastly bigger dataset. 


We wondered whether you could similarly label a small number of videos from different contexts (e.g, human video) with sentences, then force the latent trajectory space to be structured around these so that similar behaviour across contexts is embedded in the same parts of the trajectory space. For example, draw opening is more similar across different drawers than block stacking is - and the language labels would allow you to define this. Corey advised that this could well work, and that their team had been interested in exploring effectively the same concept, by using contrastive learning, the sentence labels and pre-trained sentence embeddings to supply +ive/-ive pairs. 

Our hypothesis was that this would mean the entire architecture would be easier to transfer learn to new environments or behaviours - as the feature encoder would be trained on a diverse array of inputs, and the trajectory space would be structured around behaviour and scenes outside the teleoperation dataset. 

First step though? We had to build the tech stack up to that point - a good environment to teach a robot to play in, then reimplement LFP and LangLMP.

Before we get into it? Check out this gorgeous embedding space of trajectories.
![alt-text-1](https://sholtodouglas.github.io/images/play/latent_space.png "latent space")


# How hard can a great environment be? 

Pretty damn hard. 
