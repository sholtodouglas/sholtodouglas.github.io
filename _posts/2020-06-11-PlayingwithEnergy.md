---
layout: post
title: Playing with Energy Models
categories: [Energy Models, RL]
---


![alt-text-1](https://sholtodouglas.github.io/images/energy/energyincreasing.png "Energy Model resolution increasse with neural net size")

# Planning and Reinforcement Learning
Recently I became interested in looking at combining planning and reinforcment learning, inspired by: 
- [Hallucinative Topological Memory for Zero-Shot Visual Planning (HTM)](https://arxiv.org/pdf/2002.12336.pdf), which uses a conditional VAE to sample from possible states in the environment, measures the distance between all of them using a model trained on example expert trajectories, then uses graph search to find the shortest path between start state and goal. A lower level tries to reach each state along this shortest path as a series of sub-goals. The C-VAE is able to adapt to unseen environments from the same theme at test time.

- [Search on the Replay Buffer: Bridging Planning and Reinforcement Learning (SORB)](https://arxiv.org/abs/1906.05253), instead of using a conditional VAE they sample from observed states in the replay buffer, and then measure the distance between them using the value function of the lower level (by taking the value of a state conditional on the other state as a goal). Similar graph search and lower level subgoal reaching follows. 

I like these ideas as they break down long horizon problems into achievable subgoals and are highly interpretable. I prefer the use of the q or value functions from the lower level to measure distance between states as per SORB, as this does not require expert trajectories. However, I find the idea of randomly sampling from the replay buffer to find potential subgoals inefficient - and would prefer to only search on states likely to be between the current state and the goal. By building off the C-VAE of HTM, I thought it might be possible to create a generative model which generated states likely to be between the states as opposed to from the environment generally. This could be done by using the trajectories collected during RL exploration, and training a model to generate the states in between observed states. These will not be generated in an optimal path - but thats what the graph search is for! 




# Energy Models for Generative Modelling
Concurrently, I was inspired to look into energy models after Yann LeCun's recent [video on energy based self supervised learning](https://www.youtube.com/watch?v=A7AnCvYDQrU). These are an extremely promising form of model for generative modelling. Variational Autoencoders typically require the use of reconstruction losses - which do not perform well on more complex generative tasks like images because per pixel reconstruction error is a poor proxy for overall image quality. Generative Adversarial Nets overcome this with the learned discriminator (and thus produce much better quality generated samples), but balancing the adversarial training is extremely difficult. 

Energy models offer a way to train a single model, with self-supervised losses that has much more expressive capacity than VAEs.

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
