import pandas as pd
import numpy as np 

def dgreedy_alg(mean_matrix,iterations,change_points,decay):
    n_arms = len(mean_matrix[0])
    total_Dgreedy_set = [list() for i in range(n_arms)]
    traffic_ratio_Dgreedy=[[1/n_arms for i in range(n_arms)]]  
    xbar_list=[list() for i in range(n_arms)] 
    n_list= [list() for i in range(n_arms)]  
    choice_list=[]
    
    for i in range(iterations):
        Darms = generate_arms(i,change_points,iterations,total_backet,traffic_ratio_Dgreedy,total_Dgreedy_set,n_arms)
        
 
        choice,traffic_ratio,xbar_list,n_list = discounted_epsilon_greedy(arms=Darms,epsilon=0.01,decay=decay,xbar_list=xbar_list,n_list=n_list,ratio_greedy=traffic_ratio_Dgreedy)
        choice_list.append(choice)
        traffic_ratio_Dgreedy.append(traffic_ratio)

    return choice_list,traffic_ratio_Dgreedy

def discounted_epsilon_greedy(arms, epsilon,decay,xbar_list,n_list,ratio_greedy):
  
    last_ratio = ratio_greedy[-1]
    estimated_means=generate_discounted_ratio_list(arms,xbar_list,n_list,decay)
    best_mean = np.argmax(estimated_means)

    be_exporatory = np.random.rand() < epsilon  
    if be_exporatory:  # totally random, excluding the best_mean
        other_choice = np.random.randint(0, len(estimated_means))
        while other_choice == best_mean:
            other_choice = np.random.randint(0, len(estimated_means))
        choice = other_choice
    else:  # take the best mean
        choice = best_mean
    
    new_ratio=[[] for _ in range (n_arms)]
    for i in range(n_arms):
        if i == choice:
            new_ratio[i] = 1-0.95*(sum(last_ratio)-last_ratio[i])
        else:
            new_ratio[i] = last_ratio[i]*0.95

    return choice,new_ratio,xbar_list,n_list 
