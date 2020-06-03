---
layout: post
title: Does Hierarchial Reinforcement Learning work yet?
categories: [Hierarchial, RL]
img: images/hierarchial/pickplacewsubgoal.gif
---

Hierarchial Reinforcment Learning (HRL) carries a lot of unrealised promise. Using one model to break difficult, long time horizon goals into piecemeal, achievable goals handled by a different model should make solving tasks easier. It parallels the way we approach tasks. Our brains do not operate at the level of individual muscle fibres, but consider abstract goals which are broken into a sequence of stages that are then carried about by the motor cortex. 

Hierarchy directly addresses fundamental challenges in RL, such as credit assignment and exploration. All modern RL algorithms in continuous domains fail [as the time resolution approaches zero](https://openai.com/blog/ingredients-for-robotics-research/) because it becomes impossible to discern which actions in which states led to the positive or negative outcomes. Similarly, structuring and correlating exploration [is critical](https://arxiv.org/pdf/1802.07245.pdf), which is more difficult as temporal extent increases. 

Despite this, it has largely failed to deliver on the hype. No major results use HRL, and recent work has found that even state of the art HRL algorithms amount to better exploration schemes. Indeed, OpenAI commented following their landmark DOTA results, that hierarchy simply proved unnecessary. 
> "RL researchers (including ourselves) have generally believed that long time horizons would require fundamentally new advances, such as hierarchical reinforcement learning. Our results suggest that we haven’t been giving today’s algorithms enough credit — at least when they’re run at sufficient scale and with a reasonable way of exploring."

Two pieces of work made me interested in exploring whether this was about to change. 
- [Relay Policy Learning (RPL) ](https://relay-policy-learning.github.io/), by Gupta et al achieved state of the art robotic manipulation results by training a two layer hierarchy with behavioural cloning fine-tuned with RL. 

- [Learning Multi-Level Hierarchies with Hindsight (HAC)](https://arxiv.org/pdf/1712.00948.pdf), by Levy et al is an elegant approach to training hierarchial models based on hindsight experience replay (HER) which solves the issue of non stationary lower levels. They achieve results on a set of simple problems that exceed non-hierarchial models. 

I decided to try extend RPL by using off policy, hindsight based learning like because this should be significantly more sample efficient than the on policy learning used in RPL and potentially make it viable for real world robotics.

This blog post uses a [simple test environment](https://github.com/sholtodouglas/pointMass) where a pointmass must push a block to a target position. This is an ideal testing environment because it is fast to train but contains basic versions of the difficulties facing robotic manipulation tasks (namely, that working out how to even manipulate the block requires significant exploration of the environment). Unfortunately, I failed to succeed at more complex environments, such as the same task but with multiple blocks and a robotic manipulation environment, but hey, [RL is hard](https://www.alexirpan.com/2018/02/14/rl-hard.html). The most likely issue is that the off-policy HRL framework I am using is too unstable compared to the on-policy algorithm used in RPL. With the recent release of the [architecture specifics](http://proceedings.mlr.press/v100/gupta20a/gupta20a.pdf) and [simulation environment](https://github.com/google-research/relay-policy-learning) of RPL I plan to revisit this with that in mind. 
 
# What is Hierarchial Reinforcement Learning? 

[This is an excellent explaination](https://thegradient.pub/the-promise-of-hierarchical-reinforcement-learning/). 
As a quick refresher, the standard formulation of RL involves an environment with state s<sub>t</sub>  and a policy which outputs  action a<sub>t</sub> = Pi(s<sub>t</sub>) at each timestep. In response to the action, the environment transitions s<sub>t+1</sub>,  r<sub>t+1</sub> = f(s<sub>t</sub> ,a<sub>t</sub> ), outputting a new state and a scalar reward value.  

Goal conditioned RL extends this by introducing a goal state (or subset of the state). The policy now acts based on the goal a<sub>t</sub> = Pi(s<sub>t</sub>, s<sub>g</sub>), and the reward function depends on the goal r = R(s<sub>t</sub>, s<sub>g</sub>). 

HRL typically adapts this goal conditioned framework. In the 2 layer case, this means that the higher model, which acts at a lower time frequency, will take as input the current and goal states and output as its action an easier to achieve goal on the path to that goal (a sub goal). The lower model, which acts at the full time frequency and in the action space of the environment, takes the sub goal as the input goal and attempts to reach that without a knowledge of the end goal. This means that one model is responsible for partitioning a task into subgoals, and another for solving each sub goal.   


# Analysing Learning Multi-Level Hierarchies with Hindsight (HAC)

HAC extends Hindsight Experience Replay (HER) to the hierarchial setting.

HER allows goal conditioned sparse reward environments to be efficiently solved. Sparse rewards give 0 for success and a -1 for failure, while dense rewards resemble a continuous function, such as euclidean distance between the current state and the goal state. Sparse rewards are preferred to dense rewards because dense rewards can contain local minima and require time consuming design. However, dense rewards guide an agent gradually through the environment, unlike with sparse rewards where an agent only receives signal when it succeeds, meaning that at first it randomly fails until it accidentally succeeds. 

HER provides more signal by replacing the desired goal with the goal achieved at the end of each trajectory. This means that even if the agent never achieves the desired goal in any trajectory (such as placing a block in the commanded position), instead of only learning how not to succeed - it learns how to succeed at other potential goals (by knocking the block into them accidentally). This becomes a natural curriculum that leads the agent to learn how to succeed at more and more precise goals. 

HAC uses hindsight of both the action and the goal to train each level of the hierarchy independently. If the lower levels are unable to achieve the sub goals which are set, then even if the higher level is setting the correct sub goals, it will not reach it's own goal and thus learn that the sub goals it set (i.e the action it took) are incorrect to reach that goal. Furthermore, because the lower levels are still training, the states reached will always be different and the higher level will be unable to learn any relation between the subgoals it outputs and the behaviour of the lower levels. To solve this, in the (s, a, s', r,s_g) tuple recorded by the replay buffer of the higher level, HAC substitutes the action (i.e, the subgoal commanded) with the state that was actually achieved by the lower level. This means that the model is always training as though the lower level is perfect at achieving the goals set, and learns the correct relationship between goals set and progress toward the real goal.

This is a more elegant approach than their [original idea](https://arxiv.org/pdf/1712.00948.pdf), which was to penalise the higher level for setting unreachable subgoals, and may have been inspired by the [HER+HRL request for research](https://openai.com/blog/ingredients-for-robotics-research/). As they found and we will confirm, a small amount of this 'subgoal testing' is still required, as while the model should learn the correct value for states that are ultimately reachable by the lower level, the model can overestimate the value of genuinely unreachable goals (for example, coordinates within walls), because substituting the achieved goal for the higher level action means it would appear they are never commanded. 

So, in making this work we are interested in a couple of questions.

- What time horizon works best for the higher level to reset the subgoal?
- Is subgoal testing required?
- What is the higher level learning?
- What kind of sub goal performs best? Is it the full state of the environment, just the goal relevant dimensions, or just the directly controllable dimensions corresponding to the agent itself? 
- Finally, does hierarchy provide benefits beyond better exploration?


### Environment and Algorithm Details
All models are trained with Soft Actor Critic and Hindsight experience replay for a fair comparison between hierarchial and nonhierarchial models. We only investigate two level hierarchies, as RPL found this sufficient for complex manipulation. 

## Time Horizon

How often should the higher level reset the subgoal which the lower level is trying to reach? If it is every timestep, then this eliminates the expected advantages of hierarchy, but if it is too infreqent, then the model may not adapt to new circumstances effectively or explore diversely enough within each trajectory. On this problem, a new subgoal every 5 timesteps appears to be ideal, learning 2-3x as fast as a nonhierarchial baseline.

![alt text](https://sholtodouglas.github.io/images/hierarchial/hiervsnot.png "Hierarchy vs Single Layer")


# What kind of Sub goal performs best?

On the left, subgoals in the full environment state are visualised - with the transparent pointmass and block visualising the subgoal. On the right, subgoals exclusively in the controllable dimensions of the environment are visulised - only the pointmass itself. In the center, a non hierarchial model without subgoals is shown. 

If the subgoal is exclusively pointmass position, then the lower level should learn extremely quickly as this is an easy task. However, this has the disadvantage that the lower level is not considering the intended position of the block as it acts. By including block position in the subgoal, you avoid this issue but make the lower level's task significantly more complex. 

I found that a subgoal consisting exclusively of the pointmass gave benefits to hierarchy, while a full state subgoal (or a subgoal including only the block position and not the mass positioin) was worse than solving the task non-hierarchially. In fact, the full state subgoal on the left had to be trained using relay learning - where it was the best performer. 

![alt text](https://sholtodouglas.github.io/images/hierarchial/workingcomparison.gif "Hierarchy vs Single Layer")

## Effects Learnt by the Higher Level

When using just the next position of the pointmass as a subgoal, we can visualise the expected value of every possible next position at any given state by passing them into the Q function (i.e the critic, Q(s,a)) of the higher level model. This lets us clearly visualise that the higher level is learning the correct behaviour. It assigns high value to the opposite side of the block to the goal, and low value around the block once it has placed it into the goal position.

![alt-text-1](https://sholtodouglas.github.io/images/hierarchial/qviz1.gif "title-1") ![alt-text-2](https://sholtodouglas.github.io/images/hierarchial/qviz2.gif "title-2")


## Sub Goal Testing 
 
As described earlier, by substituting the lower level achieved goal for the subgoal (higher level action) in the replay buffer, the higher level trains as though the lower level perfectly achieves the subgoals it commands. However, this will lead to the higher level assigning unknown value to states which are never reached, even if they are commanded. Levy et al's solution is to periodically set the lower level policy to deterministic instead of stochastic, not substitute the subgoal for the achieved goal and then assign a large negative reward to transitions where the higher level policy sets unreached subgoals. I wondered if instead, it would be sufficient to simply not subsitute subgoal for achieved goal some fraction of the time, so that the model would learn that unreached goals do not progress it towards the goal. 

As expected, if we never keep the original action and don't directly penalise unreached subgoals, the model proposes impossible subgoals because each time it sets one, the replay buffer only sees the goal that was actually reached - not the proposed impossible goal. This totally fails to learn. 

![alt text](https://sholtodouglas.github.io/images/hierarchial/goalsfaraway.png "Hierarchy vs Single Layer")

Surprisingly, subgoal testing with the achieved goal instead of the original goal performs best. This is truly bizzare, and actually arose from a bug in my code where I was still subsituting the action when subgoal testing. What should perform best is subgoal testing, but with the action that was unreachable kept and penalized, instead of subsituted. 

![alt text](https://sholtodouglas.github.io/images/hierarchial/sgtestingvsnot.png "Hierarchy vs Single Layer")




## Benefits beyond Exploration

Ofir et al, in [Why does Hierarchy (Sometimes) Work So Well in Reinforcement Learning](https://arxiv.org/pdf/1909.10618.pdf), train a 'shadow learner', a single layer policy trained on the transitions collected by the hierarchial policy (using their algorithm HIRO). They do this to disentangle the benefits of HRL for exploration and modelling capacity. My results here match theirs - there is no significant difference when the models are trained with the same transitions. As it stands, the benefit of HRL is in exploration. 

![alt text](https://sholtodouglas.github.io/images/hierarchial/benefitsofexplorationhierarchially.png "Hierarchy vs Single Layer")




# Relay Policy Learning
[Relay Policy Learning (RPL) ](https://relay-policy-learning.github.io/) uses the same two layer, goal conditioned hierarchial policy, but pretrains both layers with supervised learning based on the goal relabelling they used in [Learning from Play (LFP) ](https://arxiv.org/pdf/1903.01973.pdf). Rather than assuming demonstration data must correspond to a desired task, they interact widely with the environment and regard any state reached along these trajectories as a potential goal. 

To create higher level training data, they sample a sub-trajectory from the data and take the final state as the goal state. If the higher level acts every n steps, then observation, action pairs are simply (o_t, a_t+n) for all timesteps t throughout the trajectory. Lower level training data takes windows of size n from the trajectories, and sets the final state as the goal state, using every timestep within each window as data. 

This greatly expands the available data, because each observed state, action pair is valid for many goals. It also makes the data significantly easier to collect for a human teleoperator, as instead of having to reset the environment after each demonstration - the human can freely play. In LFP they found that models purely trained using supervised learning were able to comfortably complete a variety of robotic manipulation tasks. 

In RPL, they take the goal conditioned behavioural cloning model (GCBC) from LFP, make it hierarchial and finetune only the lower level model. They found this sufficient for very long horizon robotics tasks. They used a variant of TRPO, but here I used HAC from above, which should be both more efficient and better suited for the sparse rewards of manipulation tasks - but less stable as noted. 

The training is composed of two parts, firstly the inital pretraining using supervised learning, then training with a combination of the supervised and reinforcement learning losses. 

The pretrained baselines did learn to complete the task sometimes, but not reliably or efficiently as seen by the average reward and visualisations of the models below. 

| Subgoal Components   | Reward for Pretrained Models           |
|----------------------|----------------------------------------|
| Block                | -193.80689655172415 0.0896551724137931 |
| Point Mass           | -178.09558823529412 0.3014705882352941 |
| Point Mass and Block | -166.472                               |

![alt text](https://sholtodouglas.github.io/images/hierarchial/comparison.gif "Hierarchy vs Single Layer")

By introducing the HAC finetuning on the lower level, our Relay learning model learns to solve the full task with equivalent performance to all other models, but in many less update steps. It is even more efficient, with 10x less expert demonstrations than behavioural cloning on a nonhierarchial model (2000 timesteps of expert demonstration, vs 20000 timesteps of expert demonstration). 

![alt text](https://sholtodouglas.github.io/images/hierarchial/final_comparison.png "Hierarchy vs Single Layer")

![alt text](https://sholtodouglas.github.io/images/hierarchial/workingcomparison.gif "Hierarchy vs Single Layer")



Step 1

Non stationarity means issues that higher levels can typically only converge once the lower level learns to complete the higher level's instructions. 

, the practise of using models at different timescales (for example, one which considers long term goals and sets short term goals, and another which carries out the short term goals)











![alt-text-1](https://sholtodouglas.github.io/images/hierarchial/pickplacewsubgoal.gif "title-1") 


![alt-text-1](https://sholtodouglas.github.io/images/hierarchial/HACworks.gif "title-1") 
//![alt-text-2](https://sholtodouglas.github.io/images/hierarchial/HACworks2.gif "title-2")

