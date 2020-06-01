---
layout: post
title: Dissecting Hierarchial Reinforcement Learning
categories: [Hierarchial, RL]
img: images/lin_reps/hierarchy_header.png
---

Hierarchial Reinforcment Learning (HRL), the practise of using models at different timescales (for example, one which considers long term goals and sets short term goals, and another which carries out the short term goals) carries a lot of unrealised promise. Intuitively, it makes sense. Breaking long time horizon, difficult to achieve goals into piecemeal, easily achieved goals should make makes solving tasks quicker and easier. It parallels the way we approach tasks. Our brains do not operate at the level of individual muscle fibres, but consider abstract goals which are broken into a sequence of stages that are then carried about by the motor cortex. Finally, it directly addresses fundamental challenges in reinforcement learning, such as credit assignment and exploration. All modern RL algorithms will fail as the time resolution approaches infinitely small (cite openai rfr) because it becomes impossible to discern which actions in which states led to the positive or negative outcomes. Similarly, exploring in this regime is difficult because it is hard to correlate exploration over time, which is critical to fully exploring the space (cite MAESN).

Despite this, it has largely failed to deliver on the hype. No major results use HRL, and recent work has found that even state of the art HRL algorithms effectively amount to better exploration schemes, the model architecture does not give any extra expressive power. Indeed, OpenAI commented following their landmark DOTA results (an exceedingly long horizon task), thathierarchy simply proved unnecessary  "RL researchers (including ourselves) have generally believed that long time horizons would require fundamentally new advances, such as hierarchical reinforcement learning. Our results suggest that we haven’t been giving today’s algorithms enough credit — at least when they’re run at sufficient scale and with a reasonable way of exploring."

Quite recently, two pieces of work made me interested in exploring whether this was about to change. [Relay Policy Learning (RPL) ] (https://relay-policy-learning.github.io/) achieved state of the art robotic manipulation results by training a two layer hierarchy using behavioural cloning fine tuned with reinforcement learning on the lower level, [Learning Multi-Level Hierarchies with Hindsight (HAC)] (https://arxiv.org/pdf/1712.00948.pdf) displays an extremely elegant approach to training hierarchial models based on hindsight experience replay (HER) which avoids the issue of 'non stationarity'. Non stationarity means issues that higher levels can typically only converge once the lower level learns to complete the higher level's instructions. They achieve results on a set of simple problems that exceed single layer benchmarks. 

I decided to try to reimplement both of these papers, but extend RPL by using off policy, hindsight based learning like Levy et al because this should be significantly more sample efficient than the on policy learning used in RPL. Additionally, sparse rewards perform significantly better than dense rewards in manipulation based goal completion tasks, which would further add to the sample complexity if we did not use hindsight. 




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
