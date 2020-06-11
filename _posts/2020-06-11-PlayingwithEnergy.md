---
layout: post
title: Playing with Energy Models
categories: [Energy Models, RL]
---


![alt-text-1](https://sholtodouglas.github.io/images/energy/energyincreasing.png "Energy Model resolution increasse with neural net size")

Recently I became interested in looking at combining planning and reinforcment learning, inspired by: 
- [Hallucinative Topological Memory for Zero-Shot Visual Planning](https://arxiv.org/pdf/2002.12336.pdf), which uses a conditional VAE to sample from possible states in the environment, measures the distance between all of them using a model trained on example trajectories, then uses graph search to find the shortest path between start state and goal. A lower level tries to reach each state along this shortest path as a series of sub-goals. 
-[Search on the Replay Buffer: Bridging Planning and Reinforcement Learning](https://arxiv.org/abs/1906.05253), instead of using a conditional VAE they sample from observed states in the replay buffer, measure the distance between them using the value function of the lower level, and then use graph search to find the shortest path. The policy of the lower level then tries to reach each sub-goal. 


Concurrently, I was inspired to look into energy models after Yann LeCun's recent [video on energy based self supervised learning](https://www.youtube.com/watch?v=A7AnCvYDQrU). 

Scalar Valued Energy Function F(x,y)

Low energy: y i s a good prediction from x 
High energy: y is a bad prediction from x 
Inference $ \hat{y} = \min_{y} F(x,y) $

In other words, given some x, lets you predict y which would be likely given x (as determined by your dataset). You can either have conditional or unconditional 

E.g, given a set of points on a spiral (with x,y coordinates), make it so points on spiral are low energy, and points off spiral are high energy. Then, given random points we can descend down the energy function and generate a spiral! 

Typically, people have done this with models which output a probability distribution. Yann Argues against this by saying that the "distribution would be infinity on the manifold, and 0 just outside it", which will lead to infinite weights.  which will give a 'golf course'. What we want is a really smooth manifold, probabilistic methods break that.

The energy is akin to learning the negative log liklihood - but he is emphatic that what we don't want to learn is a probability distribution. 

Contrastive and architectural methods. Contrastive F(x_i , y_i) is strictly smaller than F(x_i, y) for y not in the training examples. Similar to max-likilihood if you had a tractable  Architectural is methods like PCA, K-means - or his favourite which is to use a regularization term that measures the volume of space that has low energy. Sparse coding, sparse autoencoder. 

Latent Variables allow the EBM to predict multimodally, as you can sample from the latent variable while the EBM remains unprobabilistic. 

Contrastive Methods 
- Advantage, no pixel level reconstruction (unlike a conditional VAE). Difficulty, hard negative mining. 
