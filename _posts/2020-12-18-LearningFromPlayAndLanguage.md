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

We were inspired to do this by a pair of papers, (Learning from play (LFP))[https://learning-from-play.github.io/] and (Grounding language in play (LangLMP))[https://arxiv.org/abs/2005.07648]. These present a couple of important ideas:
- In a long dataset of teleoperated 'play', you can use sample any subsequence as training data for a goal conditioned policy which learns to go from arbitrary A to B. This is vastly more data efficient and effective than any other form of imitation learning we've seen, a few hours of teleoperated 'play' provides millions of overlapping subsequences. 
- These trajectories are unlikely to be optimal, they are only one of the many ways of getting from A-B. This multimodality is captured by using a seq-seq VAE as the actor. The encoder of the VAE encodes the input subsequence as a latent 'trajectory' vector $E(z| a_0,s_0 .. a_T, s_T) $, which the decoder combines with the state and goal state at each timestep to choose an action $\pi(a|z,s_t, s_g)$. 

