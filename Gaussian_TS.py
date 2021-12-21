
import pandas as pd
import numpy as np 

def get_post_nig_original(xbar,xvar,n):
    """
    Thompson Sampling更新后验

    """
    mu =0
    nu = 1
    alpha = 1
    beta = 1
    p_mu = (nu * mu + n * xbar)/(nu + n)
    p_nu = nu + n
    p_alpha = alpha + n/2
    p_beta = beta + n*xvar/2 + n * nu / (n + nu) * (xbar - mu)**2/2


    return p_mu,p_nu,p_alpha,p_beta
def monte_carlo_simulation(arms,draw_frequency=10000):
    """
    Thompson Sampling进行蒙特卡洛采样

    input:   
    arms - 臂的信息
    draw_frequency - 采样的次数

    output:
    TSnew_traffic_ratio - 流量分配

    """
    
    mc_TS = np.zeros((draw_frequency, len(arms)))
    for i in range(len(arms)):
        TSarm = arms[i]
        post_mu_TS, post_nu_TS, post_alpha_TS, post_beta_TS = get_post_nig_original(TSarm .get('findex'), TSarm .get('fvar'), int(TSarm .get('deno')))
        mc_TS[:,i] = rnig_mu_marg(post_mu_TS, post_nu_TS, post_alpha_TS, post_beta_TS, n = draw_frequency)

    best_count_each_arm_TS = [0 for _ in arms]
    
    winner_idxs_TS = np.array(mc_TS.argmax(axis=1)).reshape(draw_frequency, )
    

    for idx2 in winner_idxs_TS:
        best_count_each_arm_TS[idx2] += 1
    
    tr_TS = [float(count) / draw_frequency for count in best_count_each_arm_TS]

    TSnew_traffic_ratio=[]
    TSMAX_RATIO = max(tr_TS)
    for item in tr_TS:
        if item == TSMAX_RATIO:
            TSnew_traffic_ratio.append(0.99*TSMAX_RATIO)
        else:
            TSnew_traffic_ratio.append((1-0.99)/(len(arms)-1)*TSMAX_RATIO+item)

    return TSnew_traffic_ratio
def TS_alg(mean_matrix,iterations,change_points):
    n_arms = len(mean_matrix[0])
    traffic_ratio_TS=[[1/n_arms for i in range(n_arms)]]
    total_TS_set = [list() for i in range(n_arms)]

    for i in range(iterations):
        arms = generate_arms(i,change_points,iterations,total_backet,traffic_ratio_TS,total_TS_set,n_arms)

        traffic_ratio= monte_carlo_simulation(arms=arms)
        traffic_ratio_TS.append(traffic_ratio)
    
    TS_results=[]
    for i in range(len(traffic_ratio_TS)):
        TS_results.append(np.argmax(traffic_ratio_TS[i]))
    
    return TS_results,traffic_ratio_TS