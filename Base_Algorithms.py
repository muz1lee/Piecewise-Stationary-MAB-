

import numpy as np
import pandas as pd
from scipy.stats import lognorm
from scipy.stats import t as stat_t
from scipy.stats import norminvgauss

import pandas as pd

def Gaussian(mean,number):
    """
    根据均值生成数据
    mean - 模拟的臂的均值
    number - 生成数据的数量
    """
    return list(np.random.normal(mean,abs(mean*(1-mean)),number)) # 方差采用p*(1-p)进行计算
def generate_arms(i,change_points,iterations,total_backet,traffic_ratio,total_set,n_arms):

    # 该轮总曝光数*每一个arm的流量占比 = 每一个arm的流量人数
    DgreedyN_list = [int(total_backet*traffic_ratio[-1][i]) for i in range(n_arms)]
    
    # 根据不同的阶段生成不同的均值数据
    if i<=change_points[0]*iterations:
        Gaussianlist =[Gaussian(mean_matrix[0][j],DgreedyN_list[j]) for j in range(len(DgreedyN_list))]

    elif i<=change_points[1]*iterations:
        Gaussianlist =[Gaussian(mean_matrix[1][j],DgreedyN_list[j]) for j in range(len(DgreedyN_list))]
        
    else:
        Gaussianlist =[Gaussian(mean_matrix[2][j],DgreedyN_list[j]) for j in range(len(DgreedyN_list))]    

    for i in range(n_arms): # number of arms
        total_set[i].extend(item for item in Gaussianlist[i] )
    
    #构建Discounted greedy算法下arm的list
    arms=[]
    temp={}
    for i in range(len(total_set)):
        temp = {
            # "previous_findex":previous_DTS_mean[i],
            "deno":len(total_set[i]), #这个是累计的曝光数量
            "findex":np.array(total_set[i]).mean(),
            "fvar":np.array(total_set[i]).var()
                }
        arms.append(temp) 

    return arms
def calculate_regret(algorithm_results,actual_best,mean_matrix,change_points,iterations):
    algorithm_hits=[0]
    regret_list=[0]
    for i in range(len(actual_best)-1):
        if algorithm_results[i]==actual_best[i+1]:
            regret=0
            algorithm_hits.append(1)
        else : 
            algorithm_hits.append(0)
            if i<=change_points[0]*iterations:
                regret = mean_matrix[0][actual_best[i+1]] - mean_matrix[0][algorithm_results[i]]

            elif i<=change_points[1]*iterations:  
                regret = mean_matrix[1][actual_best[i+1]] - mean_matrix[1][algorithm_results[i]]

            else: 
                regret = mean_matrix[2][actual_best[i+1]] - mean_matrix[2][algorithm_results[i]]
        
        #regret = 1-traffic_ratio_DTS[i+1][actual_best[i+1]]
        regret_list.append(regret)
    return algorithm_hits,regret_list
def save_result(hits_results,regret_results,traffic_ratio_results,hits_list,item,results,hits,ratio,regret):
    hits_list[str(item)] = results
    hits_results[str(item)] = hits
    traffic_ratio_results[str(item)]=ratio[:-1]
    regret_results[str(item)]=regret

    return hits_results,regret_results,traffic_ratio_results,hits_list
def generate_discounted_ratio_list(arms,xbar_list,n_list,decay):
    if len(xbar_list[0])==0:
        xbar_list =  [  [ float( arms[i].get('findex') ) ] for i in range(len(arms))  ]  
        n_list =   [  [ float(arms[i].get('deno') ) ]  for i in range(len(arms)) ] 

        estimated_means = xbar_list
    else:
        for i in range(len(arms)):
            # 数据处理，得到每一轮的均值和每一轮的曝光，然后存储在xbar_lisr和n_list
            xbar_list[i].append(  (arms[i].get('findex')*arms[i].get('deno') - sum([  xbar_list[i][j]*n_list[i][j] for j in range(len(n_list[i]))  ]))/(arms[i].get('deno')-sum(n_list[i]))     )
            n_list[i].append( arms[i].get('deno')-sum(n_list[i]) )

        #  得到discounted rate 的list        
        decay_list=[[xbar_list[i][j]*decay**(len(xbar_list[i])-1-j) for j in range(len(xbar_list[i]))] for i in range(len(arms))]

        #  得到discounted rate * 曝光数的list
        decay_number_list=[[n_list[i][j]*decay**(len(n_list[i])-1-j) for j in range(len(n_list[i]))] for i in range(len(arms))]

        # 通过公式对每一个臂的均值进行再计算
        estimated_means = [ sum([decay_list[i][j]*decay_number_list[i][j]  for j in range(len(decay_list[i]))])/sum(decay_number_list[i])  for i in range(len (arms) )]

    return estimated_means 
def rnig_mu_marg(mu,nu,alpha,beta,n=1):
    y = stat_t.rvs(df=2*alpha,size=n)
    m = np.sqrt(beta/(nu*alpha))*y+mu
    return m    