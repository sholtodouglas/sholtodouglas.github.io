---
layout: post
title: Linearised State Representations for Reinforcement Learning
categories: Miscellaneous
img: images/lin_reps/hallucination.png
custom_url: linearised_state_representations
---

Recently I found a way to learn state representations such that linear interpolation between the latent representations of states provided near optimal trajectories between the states in the original set of dimensions. They are learnt by optimising the representations of expert trajectories to lie along lines in higher dimensional latent space. The hard problem of finding the best path between states is reduced to the simple problem of taking a straight line between the latent representations of states - and the complexity is wrapped in the mapping to and from the latent space. This even worked for image based object manipulation tasks, and might be an interesting way to approach sub-goal generation for temporally extended manipulation tasks or provide a dense reward metric where latent Euclidean distance is a ‘true’ measure of progress towards the goal state.

While I transfer over to this blog, [here is the old version](https://sholtodouglas.github.io/linearised_state_representations/).
