import pandas as pd
import numpy as np 
def dgreedy_alg(mean_matrix,iterations,change_points,decay):
    """
    Discounted Epsilon greedy算法

    input:
    df - 一个实验的数据
    decay - 传入的discounted rate
    
    output:
    choice_list-算法每一轮选臂的情况
    traffic_ratio_Dgreedy- 算法每一轮选臂所分配的流量占比
    traffic_number_Dgreedy-算法每一轮选臂所分配的流量人数

    """
    n_arms = len(mean_matrix[0])
    total_Dgreedy_set = [list() for i in range(n_arms)]
    traffic_ratio_Dgreedy=[[1/n_arms for i in range(n_arms)]]  # 用于存储流量分配的占比,第一轮流量为均分，所以直接在list存储 1/n_arms
    xbar_list=[list() for i in range(n_arms)] # 将累计均值处理为每一轮的均值
    n_list= [list() for i in range(n_arms)]  # 将累计曝光数处理为每一轮的曝光数
    choice_list=[]
    
    for i in range(iterations):
        Darms = generate_arms(i,change_points,iterations,total_backet,traffic_ratio_Dgreedy,total_Dgreedy_set,n_arms)
        
        # 调用函数 discounted_epsilon_greedy,传入Darm的参数，超参epsilon, 每一轮的均值和曝光数，每一轮的分配流量占比（第一轮为均分流量）,得到选择的arm和流量分配，和更新的均值list和曝光list
        choice,traffic_ratio,xbar_list,n_list = discounted_epsilon_greedy(arms=Darms,epsilon=0.01,decay=decay,xbar_list=xbar_list,n_list=n_list,ratio_greedy=traffic_ratio_Dgreedy)
        choice_list.append(choice)
        traffic_ratio_Dgreedy.append(traffic_ratio)

    return choice_list,traffic_ratio_Dgreedy# 返回累积的数据(选择的臂,流量的分配)
def discounted_epsilon_greedy(arms, epsilon,decay,xbar_list,n_list,ratio_greedy):
    """
    Discounted Epsilon greedy算法分配流量和选臂的具体过程
    函数 dgreedy_alg将会调用函数discounted_epsilon_greedy

    input:
    arms- 实验中的臂
    epsilon- 超参，在这里我设置的0.01
    ratio_greedy - 流量占比
    
    output:
    choice: 根据当前这一轮选择下一轮pull的臂
    new_ratio - 调整新的流量
    xbar_list -  是一个列表，保存了每一轮的均值
    xvar_list - 是一个列表，保存了每一轮的方差
    """
    last_ratio = ratio_greedy[-1]
    estimated_means=generate_discounted_ratio_list(arms,xbar_list,n_list,decay)
    best_mean = np.argmax(estimated_means)

    be_exporatory = np.random.rand() < epsilon  # should we explore?
    
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
