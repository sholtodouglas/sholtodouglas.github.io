---
layout: post
title: Linearised State Representations for Reinforcement Learning
categories: Miscellaneous
img: images/lin_rep/hallucination.png
---

Recently I found a way to learn state representations such that linear interpolation between the latent representations of states provided near optimal trajectories between the states in the original set of dimensions. They are learnt by optimising the representations of expert trajectories to lie along lines in higher dimensional latent space. The hard problem of finding the best path between states is reduced to the simple problem of taking a straight line between the latent representations of states - and the complexity is wrapped in the mapping to and from the latent space. This even worked for image based object manipulation tasks, and might be an interesting way to approach sub-goal generation for temporally extended manipulation tasks or provide a dense reward metric where latent Euclidean distance is a ‘true’ measure of progress towards the goal state.

## Some great heading (h2)

Proin convallis mi ac felis pharetra aliquam. Curabitur dignissim accumsan rutrum. In arcu magna, aliquet vel pretium et, molestie et arcu.

Mauris lobortis nulla et felis ullamcorper bibendum. Phasellus et hendrerit mauris. Proin eget nibh a massa vestibulum pretium. Suspendisse eu nisl a ante aliquet bibendum quis a nunc. Praesent varius interdum vehicula. Aenean risus libero, placerat at vestibulum eget, ultricies eu enim. Praesent nulla tortor, malesuada adipiscing adipiscing sollicitudin, adipiscing eget est.

## Another great heading (h2)
