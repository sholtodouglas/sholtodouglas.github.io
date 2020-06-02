---
layout: post
title: Dissecting Hierarchial Reinforcement Learning
categories: [Hierarchial, RL]
img: images/lin_reps/hierarchy_header.png
---

Hierarchial Reinforcment Learning (HRL) carries a lot of unrealised promise. Intuitively, it makes sense. Using one model to break difficult, long time horizon goals into piecemeal, achievable goals handled by a different model should make solving tasks easier. It parallels the way we approach tasks. Our brains do not operate at the level of individual muscle fibres, but consider abstract goals which are broken into a sequence of stages that are then carried about by the motor cortex. Finally, it directly addresses fundamental challenges in RL, such as credit assignment and exploration. All modern RL algorithms in continuous domains fail [as the time resolution approaches zero](https://openai.com/blog/ingredients-for-robotics-research/) because it becomes impossible to discern which actions in which states led to the positive or negative outcomes. Similarly, exploring in this regime is difficult because it is hard to correlate exploration over time, which is critical to fully exploring the space (cite MAESN).

Despite this, it has largely failed to deliver on the hype. No major results use HRL, and recent work has found that even state of the art HRL algorithms amount to better exploration schemes. Indeed, OpenAI commented following their landmark DOTA results, that hierarchy simply proved unnecessary. 
> "RL researchers (including ourselves) have generally believed that long time horizons would require fundamentally new advances, such as hierarchical reinforcement learning. Our results suggest that we haven’t been giving today’s algorithms enough credit — at least when they’re run at sufficient scale and with a reasonable way of exploring."

Two pieces of work made me interested in exploring whether this was about to change. 
- [Relay Policy Learning (RPL) ](https://relay-policy-learning.github.io/) achieved state of the art robotic manipulation results by training a two layer hierarchy with behavioural cloning finetuned with RL. 
- [Learning Multi-Level Hierarchies with Hindsight (HAC)](https://arxiv.org/pdf/1712.00948.pdf) is an elegant approach to training hierarchial models based on hindsight experience replay (HER) which solves the issue of a non stationary lower level. They achieve results on a set of simple problems that exceed single layer benchmarks. 

I decided to try extend RPL by using off policy, hindsight based learning like Levy et al because this should be significantly more sample efficient than the on policy learning used in RPL, especially considering the sparse reward function.

This post is broken into a few parts. Firstly, reimplementing HAC and verifying that hierarchy does improve upon a single model. I then look into what the model is really learning using Q-heatmaps, what components of the framework are critical to performance and whether HAC effectively amounts to a better exploration regime. Then, I implement RPL and use HAC as the RL component to see whether integrating HRL with goal conditioned behavioural cloning is sufficient to tackle interesting robotic manipulation tasks.

This blog post uses a simple test environment where a pointmass must push a block to a target position. This is an ideal testing environment because it is fast to train but contains basic versions of the difficulties facing robotic manipulation tasks (namely, that working out how to even manipulate the block requires significant exploration of the environment). Unfortunately, I failed to succeed at more complex environments, such as the same task but with multiple blocks and a robotic manipulation environment, but hey, [RL is hard](https://www.alexirpan.com/2018/02/14/rl-hard.html). The most likely issue is that the off-policy HRL framework I am using is too unstable compared to the on-policy algorithm used in RPL. With the recent release of the [architecture specifics](http://proceedings.mlr.press/v100/gupta20a/gupta20a.pdf) and [simulation environment](https://github.com/google-research/relay-policy-learning) of RPL I plan to revisit this with that in mind. 
 
# What is Hierarchial Reinforcement Learning? 

[This is an excellent explaination](https://thegradient.pub/the-promise-of-hierarchical-reinforcement-learning/). 

# Analysing Learning Multi-Level Hierarchies with Hindsight (HAC)



# Analysing Relay Policy Learning

The environment has sparse rewards (which significantly improves eventual accuracy of block placement)

Additionally, sparse rewards perform significantly better than dense rewards in manipulation based goal completion tasks, which would further add to the sample complexity if we did not use hindsight. THis blog also dives into whether hierarchy amounts to better exploration, and what the higher level is actually learning by analysing Q heatmaps.

Step 1

Non stationarity means issues that higher levels can typically only converge once the lower level learns to complete the higher level's instructions. 

, the practise of using models at different timescales (for example, one which considers long term goals and sets short term goals, and another which carries out the short term goals)

![alt text](https://sholtodouglas.github.io/images/hierarchial/hiervsnot.png "Hierarchy vs Single Layer")

![alt text](https://sholtodouglas.github.io/images/hierarchial/benefitsofexplorationhierarchially.png "Hierarchy vs Single Layer")


![alt text](https://sholtodouglas.github.io/images/hierarchial/goalsfaraway.png "Hierarchy vs Single Layer")

![alt text](https://sholtodouglas.github.io/images/hierarchial/sgtestingvsnot.png "Hierarchy vs Single Layer")

## Relay Pre Training

Achieved : -193.80689655172415 0.0896551724137931
Controllable : -178.09558823529412 0.3014705882352941
Full state : -166.472 0.424

![alt text](https://sholtodouglas.github.io/images/hierarchial/comparison.gif "Hierarchy vs Single Layer")

![alt-text-1](https://sholtodouglas.github.io/images/hierarchial/qviz1.gif "title-1") ![alt-text-2](https://sholtodouglas.github.io/images/hierarchial/qviz2.gif "title-2")

![alt-text-1](https://sholtodouglas.github.io/images/hierarchial/HACworks.gif "title-1") ![alt-text-2](https://sholtodouglas.github.io/images/hierarchial/HACworks2.gif "title-2")


![alt text](https://sholtodouglas.github.io/images/hierarchial/final_comparison.png "Hierarchy vs Single Layer")

![alt text](https://sholtodouglas.github.io/images/hierarchial/workingcomparison.gif "Hierarchy vs Single Layer")