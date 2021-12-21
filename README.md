# Piecewise-Stationary-MAB
This project was finished during my internship. It aims to handle with the issue of Non-Stationary MAB in recommendations.

## Background 
Multi-Arm-Bandit (MAB) is a classic problem that well demonstrates the exploration vs exploitation dilemma. This dilemma comes from the incomplete information : we need to gather enough information to make best overall decisions while keeping the risk under control.

In product design , Data Analysts will use A/B testing to test which version ( such as layout, UI design ) could obtain higher Click-Through Rates (CTR) / Conversion Rates , etc.  However, A/B testing is not suitable for the following scenarios : 

1. "Expensive to try" : For example, clicks on advertising will bring direct benefits for the company. However if too much traffic is allocated to bad advertisements, the overall click-through rate will drop, which will eventually lead to a direct decline in revenue ; 

2. Time-sensitive scenarios: In real-time news recommendation, we need find the best news as soon as possilble, or in some scenarios rewards  will change over time, and our system needs to make adjustmetns in a timely manner ;

3.Insufficient / Limited traffic : Sicne there are many experiments to conduct while traffic is limited, we need to find a more efficient way to optimize and allocate traffic 
