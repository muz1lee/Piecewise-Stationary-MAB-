import numpy as np
import pandas as pd
from Base_Algorithms import * 

def DTS_alg(mean_matrix,decay,iterations,change_points):
    n_arms = len(mean_matrix[0])
    traffic_ratio_DTS=[[1/n_arms for i in range(n_arms)]]
    total_DTS_set = [list() for i in range(n_arms)]
    xbar_list=[list() for i in range(n_arms)]
    n_list= [list() for i in range(n_arms)]
    var_list= [list() for i in range(n_arms)]
    for i in range(iterations):
        Darms = generate_arms(i,change_points,iterations,total_backet,traffic_ratio_DTS,total_DTS_set,n_arms)

        traffic_ratio_new,xbar_list,n_list,var_list = monte_carlo_simulation_DTS(Darms=Darms,decay=decay,xbar_list=xbar_list,n_list=n_list,var_list=var_list)
        traffic_ratio_DTS.append(traffic_ratio_new)

        
        DTS_results=[]
        for i in range(len(traffic_ratio_DTS)):
            DTS_results.append(np.argmax(traffic_ratio_DTS[i]))

    return DTS_results,traffic_ratio_DTS

def monte_carlo_simulation_DTS(Darms, decay,xbar_list=[list(),list(),list()],n_list=[list(),list(),list()],var_list=[list(),list(),list()],draw_frequency=10000):
    
    mc_DTS = np.zeros((draw_frequency, len(Darms)))
    for i in range(len(Darms)):
        DTSarm = Darms[i]
        post_mu_DTS, post_nu_DTS, post_alpha_DTS, post_beta_DTS,xbar_list[i],n_list[i],var_list[i] = get_post_nig_new(DTSarm.get('findex'), DTSarm.get('fvar'), int(DTSarm.get('deno')),decay,xbar_list[i],n_list[i],var_list[i])

        mc_DTS[:,i] = rnig_mu_marg(post_mu_DTS, post_nu_DTS, post_alpha_DTS, post_beta_DTS, n = draw_frequency)

    best_count_each_arm_DTS = [0 for _ in Darms]
    
    winner_idxs_DTS = np.array(mc_DTS.argmax(axis=1)).reshape(draw_frequency, )

    for idx1 in winner_idxs_DTS:
        best_count_each_arm_DTS [idx1] += 1

    
    tr_DTS = [float(count) / draw_frequency for count in best_count_each_arm_DTS]

    new_traffic_ratio=[]
    MAX_RATIO = max(tr_DTS)
    for item in tr_DTS:
        if item == MAX_RATIO:
            new_traffic_ratio.append(0.99*MAX_RATIO)
        else:
            new_traffic_ratio.append((1-0.99)/(len(Darms)-1)*MAX_RATIO+item)

    
    return new_traffic_ratio,xbar_list,n_list,var_list
def get_post_nig_new(xbar,xvar,n,decay,xbar_list,n_list,var_list):

    mu =0
    nu = 1
    alpha = 1
    beta = 1
    
    if len(xbar_list)==0:
        xbar_list.append(xbar)
        n_list.append(n)
        p_mu = xbar

    else:
 
        xbar_list.append((xbar*n - sum([xbar_list[i]*n_list[i] for i in range(len(n_list))]))/(n-sum(n_list)))
        n_list.append(n-sum(n_list))

        decay_list=[xbar_list[i]*decay**(len(xbar_list)-1-i) for i in range(len(xbar_list))]
        
        decay_number_list=[n_list[i]*decay**(len(n_list)-1-i) for i in range(len(n_list))]
        p_mu = sum (decay_list[i]*decay_number_list[i] for i in range(len(n_list)))/sum(decay_number_list)
        var_list.append( (len(var_list)+1) * xvar -sum(var_list) )
        decay_var_list=[var_list[i]*decay**(len(var_list)-1-i) for i in range(len(var_list))]
    
        p_beta = beta + n*sum(decay_var_list)/(2*len(decay_var_list)) + n * nu / (n + nu) * (p_mu - mu)**2/2
    
    p_nu = nu + n
    p_alpha = alpha + n/2

    p_beta = beta + n*xvar/2 + n * nu / (n + nu) * (p_mu - mu)**2/2
    

    return p_mu,p_nu,p_alpha,p_beta,xbar_list,n_list,var_list
