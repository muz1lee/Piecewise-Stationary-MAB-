# Piecewise-Stationary-MAB
This project was finished during my internship. It aims to handle with the issue of Non-Stationary MAB in real-time news recommendations.

## Background 
Multi-Arm-Bandit (MAB) is a classic problem that well demonstrates the exploration vs exploitation dilemma. This dilemma comes from the incomplete information : we need to gather enough information to make best overall decisions while keeping the risk under control. Comparing with A/B testing, it can handle with following issues:

1. "Expensive to try" : For example, clicks on advertising will bring direct benefits for the company. However if too much traffic is allocated to bad advertisements, the overall click-through rate will drop, which will eventually lead to a direct decline in revenue ; 

2. Time-sensitive : In real-time news recommendation, we need find the best news as soon as possilble, or in some scenarios rewards  will change over time, and our system needs to make adjustmetns in a timely manner ;

3. Insufficient / Limited traffic : Since there are many experiments to conduct while traffic is limited, we need to find a more efficient way to optimize our allocation mechanism.

## Business Requirements 

Our team want to increase the metric: CTR = Clicks / Impressions , in our Stocks News Recommendation.

Push Mechanism: push news to users from high activity to low activity sequentially, and adjust traffic every 5 minutes. For every round, the system would use MAB algorithm to explore / exploit and find a "best" arm, then allocate more traffic to this arm in next round. 


## Problems and Solutions 

Problems: 

(1) CTR > 1 : Since some users will share/forward news to others and we would obatin additional clicks in our experiments. Ratio-based metric is not suitable in our experiments.

(2) Rewards of each arm will change with time: Since users with different activity will have different preference, the rewards ( OSR ) are non-stationary for each arm. 

Solutions:

(1) Replace our metric with "Overall Sharing Rate (OSR)  = clicks/impressions" , which is a mean based metric.

(2) Introduce Non-Stationary / Piecewise-stationary MAB. Here is an assumption : The reward process of the arms is non-stationary on the whole, but stationary on intervals. This is the Piecewise-stationary Bandit Problem.

## Methods / Algorithms 
For Non-Stationary / Piecewise-stationary MAB，we have three different methods: 

(1) Reset the algorithm at suitable points

(2) Allow explicit exploration

(3) Reduce the impact of past observation

This project focus on the (3) method , implements Gaussian Thompson Sampling , Discounted Thompson Sampling, Epsilon Greedy , Discounted Epsilon Greedy, Softmax Greedy algorithms, and finally compare their performance based on regret. 



