import pandas as pd
import numpy as np 

def epsilon_greedy(arms, epsilon,ratio_greedy):

    n_arms = len(arms)
    last_ratio = ratio_greedy[-1] 
    estimated_means =[arms[i].get('findex') for i in range(len(arms))] 
    best_mean = np.argmax(estimated_means) 
    
    be_exporatory = np.random.rand() < epsilon 
    if be_exporatory:  
        other_choice = np.random.randint(0, len(estimated_means))
        while other_choice == best_mean: 
            other_choice = np.random.randint(0, len(estimated_means))
        choice = other_choice
    else: 
        choice = best_mean 
    new_ratio=[[] for _ in range (n_arms)]

    for i in range(n_arms):
        if i == choice:
            new_ratio[i] = 1-0.95*(sum(last_ratio)-last_ratio[i])
        else:
            new_ratio[i] = last_ratio[i]*0.95
    
    return choice,new_ratio
def greedy_alg(mean_matrix,iterations,change_points):
    n_arms = len(mean_matrix[0])
    traffic_ratio_greedy=[[1/n_arms for i in range(n_arms)]]
    total_greedy_set = [list() for i in range(n_arms)]
    greedy_results=[]
    for i in range(iterations):
        greedyarms = generate_arms(i,change_points,iterations,total_backet,traffic_ratio_greedy,total_greedy_set,n_arms)
        choice_arm,traffic_ratio= epsilon_greedy(arms=greedyarms,epsilon=0.01,ratio_greedy=traffic_ratio_greedy)

    
        traffic_ratio_greedy.append(traffic_ratio)
    #print(len(traffic_ratio_greedy))

        greedy_results.append(choice_arm)

    return greedy_results,traffic_ratio_greedy
