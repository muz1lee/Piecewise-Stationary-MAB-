import pandas as pd
import numpy as np 

def epsilon_greedy(arms, epsilon,ratio_greedy):

    """
    Epsilon greedy算法分配流量和选臂的具体过程
    函数 greedy_alg将会调用函数epsilon_greedy
    
    input:
    arms- 实验中的臂
    epsilon-超参，在这里我设置的0.01
    ratio_greedy -流量占比

    output:
    choice- 根据当前这一轮选择下一轮pull的臂
    new_ratio- 调整新的流量
    """
    n_arms = len(arms)
    last_ratio = ratio_greedy[-1] # 调用上一轮的流量分配结果，意在此基础上进行调整
    estimated_means =[arms[i].get('findex') for i in range(len(arms))]  #调用三个臂的均值
    best_mean = np.argmax(estimated_means) #在这三个臂中找到最好的
    
    be_exporatory = np.random.rand() < epsilon  # should we explore?
    if be_exporatory:  # totally random, excluding the best_mean
        other_choice = np.random.randint(0, len(estimated_means))
        while other_choice == best_mean: # explore， 选择非best的臂
            other_choice = np.random.randint(0, len(estimated_means))
        choice = other_choice
    else:  # 选择best arm
        choice = best_mean 
    new_ratio=[[] for _ in range (n_arms)]

    #这里是设定的流量分配方法，非choice的臂各自减少5%的流量，choice的臂则增加流量 new ratio= (95 % *last ratio of choice arm + 5% )
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