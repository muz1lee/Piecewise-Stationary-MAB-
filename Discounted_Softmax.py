import pandas as pd
import numpy as np 
# -- Discounted epsilon softmax
def monte_carlo_simulation_softmax_greedy(q_estimation, tau, epsilon=0.6, draw_frequency=10000):
    """
    softmax_greedy算法的蒙特卡洛模拟
    Args::
    ------------
    arms list[Arm]: 每个手臂的统计量
    epsilon string: 代表explore的概率
    draw_frequency int: 蒙特卡洛模拟的次数

    Returns:
    ------------
    mc np.matrix: Monte Carlo matrix of dimension (draw, n_arms).
    traffic_ratio list[float]: 各个手臂的流量比例
    """
    # tau = 100  # 目前tau的大小是根据腾讯视频推荐的场景定的，后面应作为可传入的超参
    # estimated_rewards = [arm.get('q_estimation') for arm in arms]
    best_count_each_arm = [0 for _ in q_estimation]
    mc = np.zeros(draw_frequency)
    softmax_p = [0 for _ in q_estimation]
    temp_max = np.max(q_estimation)
    softmax_arm = [np.exp((j - temp_max) / tau) for j in q_estimation]
    softmax_total = np.sum(softmax_arm)
    for i in range(len(q_estimation)):
        softmax_p[i] = softmax_arm[i] / softmax_total
    threads = np.cumsum(softmax_p)
    for i in range(draw_frequency):
        if np.random.rand() < epsilon:
            # explore的部分采用softmax
            r = np.random.rand()
            for arm_i in range(len(q_estimation)):
                if r <= threads[arm_i]:
                    arm_choose = arm_i
                    break
        else:
            arm_choose = np.argmax(q_estimation)

        best_count_each_arm[arm_choose] += 1
        mc[i] = q_estimation[arm_choose]
    return [float(count) / draw_frequency for count in best_count_each_arm], mc,arm_choose
def softmax_alg(mean_matrix,iterations,change_points):

    """
    epsilon_softmax算法


    """
    n_arms = len(mean_matrix[0])
    traffic_ratio_softmax=[[1/n_arms for i in range(n_arms)]]
    total_softmax_set = [list() for i in range(n_arms)]
    softmax_results=[] 

    for i in range(iterations):
        softmaxarms = generate_arms(i,change_points,iterations,total_backet,traffic_ratio_softmax,total_softmax_set,n_arms)
        
        choice_arm,traffic_ratio= compute_traffic_mean_epsilon_softmax(arms=softmaxarms,q_estimation=None,step_size=0.5,tau = 100, epsilon = 0.1)

        softmax_results.append(choice_arm)

        traffic_ratio_softmax.append(traffic_ratio)
    
    return softmax_results,traffic_ratio_softmax
def compute_traffic_mean_epsilon_softmax(arms,q_estimation,step_size,tau , epsilon):

    """
    epsilon_softmax算法的计算流量分配


    """
    rewards = [arm.get('findex') for arm in arms]

    if not q_estimation:
        q_estimation = [0 for _ in arms]
    else:
        q_estimation = [i for i in q_estimation]

    #q_estimation_dic = {}
    for i in range(len(arms)):
        q_estimation[i] += step_size * (rewards[i] - q_estimation[i])
        #q_estimation_dic[arms[i]['id']] = q_estimation[i]
        

    traffic_ratio_tmp, mc,arm_choose = monte_carlo_simulation_softmax_greedy(
            q_estimation=q_estimation, tau=tau, epsilon=epsilon)
    #self.result['q_estimation'] = q_estimation_dic
    return arm_choose,traffic_ratio_tmp

