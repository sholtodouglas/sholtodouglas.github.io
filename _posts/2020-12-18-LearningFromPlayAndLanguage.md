---
layout: post
title: Transfer learning from play and language [Work in Progress]
categories: [play, language, imitation, latent]
---

![alt-text-1](https://sholtodouglas.github.io/images/play/play_gif.gif "title-1")


> [Code found here](https://colab.research.google.com/github/sholtodouglas/learning_from_play/blob/master/languageXplay.ipynb). 
> All of this was done with [Tristan Frizza](https://twitter.com/TristanVtx). 

* TOC
{:toc}

# Introduction

### Inspiration
Tristan and I were inspired to ask the question

> "Can we enable fast transfer learning to new scenes or behaviours by using language to exand the available training data for imitation learning?"

We were inspired to do this by a pair of papers, [Learning from play (LFP)](https://learning-from-play.github.io/) and [Grounding language in play (LangLMP)](https://arxiv.org/abs/2005.07648). These present a couple of important ideas:
- In a long dataset of teleoperated 'play', you can sample any subsequence as training data for a goal conditioned policy which learns to go from arbitrary A to B. This is vastly more data efficient and effective than any other form of imitation learning we've seen, a few hours of teleoperated 'play' provides millions of overlapping subsequences. 
- These trajectories are unlikely to be optimal, they are only one of the many ways of getting from A-B. This multimodality is captured by using a seq-seq VAE as the actor. The encoder of the VAE encodes the subsequence as a latent 'trajectory' vector $E(z \| a_0,s_0 .. a_T, s_T) $, which the decoder combines with the state and goal state at each timestep to choose an action $\pi(a \|z,s_t, s_g)$. During training, they seek to maximise the log probability of the true actions from that sequence. At test time, a 'planner' samples a potential trajectory vector given the initial and end goal $P(z \|s_0, s_g)$. During training, the actor tries to decode that specific way of achieving the goal - at test time, the actor decodes one of the many ways of achieving it. Use images rather than states as goals by embedding the image in a lower dimensional space with a CNN as a feature encoder.
- The model can be adapted to use sentences as a goal by labelling a small (<1% of total dataset in their experiments) number of trajectories. Goal images and sentences will share the same goal embedding space because they will correspond to the same sequences of actions, which will be maximised for by the same imitation learning objective as before. This allows language to control the robot, while still learning more robust control from a vastly bigger dataset. 

### Hypothesis
We wondered whether you could similarly label a small number of videos from different contexts (e.g, human video) with sentences, then force the latent trajectory space to be structured around these so that similar behaviour across contexts is embedded in the same parts of the trajectory space. For example, draw opening is more similar across different drawers than block stacking is - and the language labels would allow you to define this. Corey advised that this could well work, and that their team had been interested in exploring effectively the same concept, by using contrastive learning, the sentence labels and pre-trained sentence embeddings to supply +ive/-ive pairs. 

Our hypothesis was that this would mean the entire architecture would be easier to transfer learn to new environments or behaviours - as the feature encoder would be trained on a diverse array of inputs, and the trajectory space would be structured around behaviour and scenes outside the teleoperation dataset. 

First step though? We had to build the tech stack up to that point - a good environment to teach a robot to play in, then reimplement LFP and LangLMP.

Before we get into it? Check out this gorgeous embedding space of trajectories.
![alt-text-1](https://sholtodouglas.github.io/images/play/latent_space.png "latent space")


# How hard can a great environment be? 

### Teleoperation
Insidiously hard. We tried re-implementing LFP for one of our senior year classes and failed due to a critical environment issue which we didn't discover till we came back to crack our 'great white whale'. Saving images of the environment took a variable amount of time, which was often long enough that it affected the frame rate of data capture. This meant that the time between actions in the dataset was variable, which meant that the time which the action was executed for was variable - decoupling the learnable link between action and outcome. We spent weeks trying to improve the algorithm itself - rather than making sure the environment was learnable. 

### Keep it simple, stupid

When we came back, we made sure to do it right. No cutting corners. 

First, learn 3DOF reaching with scripted demonstrations. Then use scripted demonstrations to learn 3DOF pick/placing. Extend that to allow orientation control...

Next issue. 

### Representing orientation
What is the best way to represent orientation of the gripper? Corey et al use RPY - we thought we could be clever, and use quaternions to avoid the discontinuities inherent in RPY. A fun fact about quaternions is that the negative of the quaternion is an equivlant representation of orientation. pyBullet (the great simulator we used), is not consistent in the representation as you move in the environment. To correct this, you have to create a small check in the environment which checks if the sign on every element of the quaternion is the negative of the previous timestep, in which case - you flip the sign and thus smooth the signal. 

Problem solved? Not quite. Quaternions must be normalised to be a valid orientation. Neural net outputs have no such constraint, besides, we still had a few discontinuities. This led us down the path of [5D and 6D rotational embeddings](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhou_On_the_Continuity_of_Rotation_Representations_in_Neural_Networks_CVPR_2019_paper.pdf)...

Hold up! We had been playing with too many satellites. Does a robot's end effector really need to go beyond +/- $\pi/2$ in any axis of rotation? Will it ever encounter a discontinuity? Not unless you're trying to imitation learn off Houdini. 

Long story short - RPY orientation control worked far better. Simple fix wins out again. 

# Next steps

The algorithm now works reasonably well - it reliably achieves complex, two step goals even without a latent trajectory space to capture the multimodality. 

We're actually struggling to train a model which uses the latent space that is better than goal conditioned behavioural cloning (GCBC) - when the space is structured enough for the decoder to be able to perfectly recreate encoded trajectories, it is too sparse for the planner to guess potential trajectories. With sufficient regularisation to be plannable, the space quickly loses structure and meaning and performance is identical to GCBC. We've converged on sufficently low Beta values (see the LFP paper) that we do observe some structure, but haven't found the optimal point. We are also encountering overfitting issues - which we hope is due to the multimodality (without a sufficiently informative trajectory vector, the model must memorise the inputs to continue to improve, the goal state is insufficient information). We hope that a well structured trajectory space overcomes this by giving the model sufficient information to recreate the input trajectory without necessitating memorisation.

In parallel, we are fixing the gripper.  Gripper loss magnitude is currently 10x higher than any other dimension in the action space. We suspect this is due to how discontinuous the teleoperated actions are, and are trying a smoother 'close' over more timesteps (as is done in LFP and LangLFP). Furthermore, there is often a trade off between gripper performance and position/orientation performance as the trajectory space is regularised. We think this is the key remaining environment fix. 
