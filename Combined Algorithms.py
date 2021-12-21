
import numpy as np

from scipy.stats import lognorm
from scipy.stats import t as stat_t
from scipy.stats import norminvgauss

import pandas as pd
total_backet = 12000


# --TS 
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


# --DTS 
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
    """
    Discounted Thompson Sampling进行蒙特卡洛采样

    input:   
    Darms - 臂的信息
    draw_frequency - 采样的次数
    ratio_decay - 防止由于均值差异大导致流量分配极端，所以将最好臂的(1-ratio_decay)的流量均分给其余非best arm的臂
    xbar_list - 上一轮的 均值list 作为输入
    n_list - 上一轮的 曝光list 作为输入
    var_list - 上一轮的 方差list 作为输入


    output:
    new_traffic_ratio - 流量分配
    xbar_list - 该轮更新的 均值list
    n_list - 该轮更新的 曝光list
    var_list - 该轮更新的 方差list
    """
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
    """
    Discounted Thompson Sampling更新后验

    """
    mu =0
    nu = 1
    alpha = 1
    beta = 1
    
    if len(xbar_list)==0:
        xbar_list.append(xbar)
        n_list.append(n)
        p_mu = xbar

    else:
        # 数据处理，得到每一轮的均值和每一轮的曝光，然后存储在xbar_lisr和n_list
        xbar_list.append((xbar*n - sum([xbar_list[i]*n_list[i] for i in range(len(n_list))]))/(n-sum(n_list)))
        n_list.append(n-sum(n_list))

        #  得到discounted rate 的list
        decay_list=[xbar_list[i]*decay**(len(xbar_list)-1-i) for i in range(len(xbar_list))]
        
        #  得到discounted rate * 曝光数的list
        decay_number_list=[n_list[i]*decay**(len(n_list)-1-i) for i in range(len(n_list))]
        p_mu = sum (decay_list[i]*decay_number_list[i] for i in range(len(n_list)))/sum(decay_number_list)
        var_list.append( (len(var_list)+1) * xvar -sum(var_list) )
        decay_var_list=[var_list[i]*decay**(len(var_list)-1-i) for i in range(len(var_list))]
    
        p_beta = beta + n*sum(decay_var_list)/(2*len(decay_var_list)) + n * nu / (n + nu) * (p_mu - mu)**2/2
    p_nu = nu + n
    p_alpha = alpha + n/2

    p_beta = beta + n*xvar/2 + n * nu / (n + nu) * (p_mu - mu)**2/2
    

    return p_mu,p_nu,p_alpha,p_beta,xbar_list,n_list,var_list

# -- 公用 -- 
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

# -- epsilon greedy 
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

# -- Discounted epsilon greedy 
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


def experiment(mean_matrix,decay_list,iterations,algorithms,change_points):
    hits_results = dict()
    regret_results = dict()
    traffic_ratio_results= dict()
    actual_best =[]
    hits_list= dict()

    for i in range(iterations):
        if i<=change_points[0]*iterations:
            actual_best.append(np.argmax(mean_matrix[0]))
        elif i<=change_points[1]*iterations:  
            actual_best.append(np.argmax(mean_matrix[1]))
        else:
            actual_best.append(np.argmax(mean_matrix[2]))

    for item in algorithms:
        print('--------------------------Begin {} algorithm--------------------------'.format(str(item)))
        if item=='DTS':
            for dr in decay_list:
                DTS_results,traffic_ratio_DTS = DTS_alg(mean_matrix,dr,iterations,change_points)
                DTS_hits,regret_list=calculate_regret(DTS_results,actual_best,mean_matrix,change_points,iterations)
                al = 'DTS_'+str(dr)
                hits_results,regret_results,traffic_ratio_results,hits_list=save_result(hits_results,regret_results,traffic_ratio_results,hits_list,al,DTS_results,DTS_hits,traffic_ratio_DTS,regret_list)
           
        if item =='TS':
            TS_results,traffic_ratio_TS = TS_alg(mean_matrix,iterations,change_points)
            TS_hits,regret_list=calculate_regret(TS_results,actual_best,mean_matrix,change_points,iterations)
            hits_results,regret_results,traffic_ratio_results,hits_list=save_result(hits_results,regret_results,traffic_ratio_results,hits_list,item,TS_results,TS_hits,traffic_ratio_TS,regret_list)

        if item=='epsilon_greedy':
            greedy_results,traffic_ratio_greedy = greedy_alg(mean_matrix,iterations,change_points)
            greedy_hits,regret_list=calculate_regret(greedy_results,actual_best,mean_matrix,change_points,iterations)
            hits_results,regret_results,traffic_ratio_results,hits_list=save_result(hits_results,regret_results,traffic_ratio_results,hits_list,item,greedy_results,greedy_hits,traffic_ratio_greedy,regret_list)


        if item=='Discounted_epsilon_greedy':
            for dr in decay_list:
                Dgreedy_results,traffic_ratio_Dgreedy = dgreedy_alg(mean_matrix,iterations,change_points,dr)
                Dgreedy_hits,regret_list=calculate_regret(Dgreedy_results,actual_best,mean_matrix,change_points,iterations)
                al = 'Dgreedy_'+str(dr)
                hits_results,regret_results,traffic_ratio_results,hits_list=save_result(hits_results,regret_results,traffic_ratio_results,hits_list,al,Dgreedy_results,Dgreedy_hits,traffic_ratio_Dgreedy,regret_list)

        if item =='epsilon_softmax':
            softmax_results,traffic_ratio_softmax = softmax_alg(mean_matrix,iterations,change_points)
            softmax_hits,softmax_results=calculate_regret(Dgreedy_results,actual_best,mean_matrix,change_points,iterations)
            hits_results,regret_results,traffic_ratio_results,hits_list=save_result(hits_results,regret_results,traffic_ratio_results,hits_list,item,softmax_results,softmax_hits,traffic_ratio_softmax,regret_list)


    return hits_results,traffic_ratio_results,actual_best,regret_list,hits_list


# def final_results_to_DataFrame(all_hits_results,all_traffic_ratio_results,all_regret_results,all_hits_list,all_actual_best):
#     final_df = pd.DataFrame(columns=['iterations', 'ratio_arm1', 'ratio_arm2', 'ratio_arm3', 'Algorithms_y', 'experiment_version','Hits_list'])

#     for i in range(len(all_traffic_ratio_results)):
#         df_traffic_ratio =  pd.DataFrame(columns=['iterations','ratio_arm1','ratio_arm2','ratio_arm3','Algorithms'])

#         for item in all_traffic_ratio_results[i].keys():
#             ratio_df = pd.DataFrame(all_traffic_ratio_results[i][item]).reset_index()
#             ratio_df.columns=['iterations','ratio_arm1','ratio_arm2','ratio_arm3']
#             ratio_df['Algorithms']=pd.DataFrame([item] for i in range(len(ratio_df)))

#             df_traffic_ratio = df_traffic_ratio.append(ratio_df)

#         df_traffic_ratio = df_traffic_ratio.reset_index().drop('index',axis=1)

#         df3 = pd.DataFrame(columns=['iteration_index','Algorithms','Hits_list'])

#         for item in all_hits_list[i].keys():
#             df1 = pd.DataFrame(all_hits_list[i][item]).reset_index()
#             df1.columns=['Algorithms','Hits_list']
#             df1['Hits_list'] = df1['Hits_list']+1
#             df1 = df1.reset_index()
#             df1['Algorithms']=item
#             df1.columns=['iteration_index','Algorithms','Hits_list']
#             df3 = pd.concat( [df3,df1],axis=0)


#         df3 = df3.reset_index().drop('index',axis=1)

#         df_merged = df_traffic_ratio.merge(df3,left_index=True,right_index=True)
#         df_merged=df_merged[['iterations', 'ratio_arm1', 'ratio_arm2', 'ratio_arm3', 'Algorithms_y','Hits_list']]

#         df_merged['experiment_version']=str(i+1)

#         final_df = final_df.append(df_merged)

#     df_total_regret = pd.DataFrame(columns=['experiment_version','iterations','algorithms','regret'])
#     for i in range(len(all_regret_results)):
#         df_regret = pd.DataFrame(columns=['algorithms','regret'])

#         df_regret_list = pd.DataFrame(all_regret_results[i])
#         for item in df_regret_list.columns:
#             new_df = df_regret_list[item].reset_index()
#             new_df.columns=['algorithms','regret']
#             new_df['algorithms']=item
#             df_regret = df_regret.append(new_df)

#         df_regret = df_regret.reset_index().reset_index()
#         df_regret.columns=['experiment_version','iterations','algorithms','regret']

#         df_regret['experiment_version']=str(i+1)

#         df_total_regret = df_total_regret.append(df_regret)


#     final_combine = final_df.merge(df_total_regret,left_on=['Algorithms_y','iterations','experiment_version'],right_on=['algorithms', 'iterations', 'experiment_version'],how='inner')
#     final_combine = final_combine[['experiment_version', 'algorithms', 'iterations','Hits_list',  'ratio_arm1', 'ratio_arm2', 'ratio_arm3','regret']]

#     binary_df = pd.DataFrame(columns=['experiment_version','iterations','Hits_or_not','algorithms'])
#     for i in range(len(all_hits_results)):
#         df_hits_binary = pd.DataFrame(all_hits_results[i])
#         for item in df_hits_binary.columns:
#             df_hb = pd.DataFrame(df_hits_binary[item]).reset_index().reset_index()
#             df_hb.columns=['experiment_version','iterations','Hits_or_not']
#             df_hb['experiment_version']=str(i+1)
#             df_hb['algorithms']=str(item)

#             binary_df = binary_df.append(df_hb)
#     binary_df.head()

#     binary_df['iterations'] = binary_df['iterations'].apply(int)
#     final_combine['iterations'] = final_combine['iterations'].apply(int)
#     binary_df = binary_df.reset_index().drop('index',axis=1)

#     final_combine = final_combine.merge(binary_df,on =['experiment_version', 'iterations', 'algorithms'])

#     final_combine.columns=['experiment_version', 'algorithms','iterations', 'Hits_list','ratio_arm1', 'ratio_arm2','ratio_arm3','regret','Hits_or_not']
#     final_combine = final_combine[['algorithms','experiment_version', 'iterations',  'ratio_arm1', 'ratio_arm2', 'ratio_arm3','Hits_or_not', 'regret']]

#     actual_df = pd.DataFrame(columns=['iterations','actual_best'])
#     for item in pd.DataFrame(all_actual_best).columns:
#         all_actual= pd.DataFrame(all_actual_best)[item].reset_index()
#         all_actual.columns=['iterations','actual_best']
#         all_actual['actual_best']=all_actual['actual_best']+1
#         all_actual['experiment_version']=str(item+1)
#         actual_df = actual_df.append(all_actual)

#     final_combine['iterations'] = final_combine['iterations'].apply(int)

#     actual_df['iterations'] = actual_df['iterations'].apply(int)

#     final_combine = pd.merge(final_combine,actual_df,how='outer',on=['experiment_version','iterations'])

#     return final_combine  


if __name__ == '__main__' :
    version_list= [0.1,0.5,1,5,10] # version list代表我们模拟的实验版本的数量
    total_backet = 12000
    decay_list = [0.99,0.9,0.8,0.7,0.6,0.5,0.3,0.1] # discounted rate list
    change_points=[0.4,0.5] # 在N次迭代中， 40% * N 和 50%*N 两个迭代点将变换均值
    iterations = 50 # 迭代的次数

    all_hits_results = dict() #保存所有arm在所有实验中的摇臂情况
    all_traffic_ratio_results = dict() #保存所有arm在所有实验中的流量分配
    all_regret_results=dict() #保存所有arm在所有实验中的遗憾值
    mean_matrix_list = dict()
    all_hits_list=dict() #保存所有arm在所有实验中是否选中最好的臂
    all_actual_best=dict() ##保存所有实验中实际最好的臂

    # There are some algorithms you could choose 
    algorithms = ['TS','DTS','Discounted_epsilon_greedy','epsilon_greedy','softmax_greedy','epsilon_softmax']

    for i in range(len(version_list)):
        print('__________________________start version {}`s training__________________________'.format(i))
        mean_matrix = np.random.rand(3,3) * version_list[i] 
        """
        # 随机生成(3,3)维度，范围为(0~1)的数字作为均值，version list的数将增大均值
        #  row = 阶段数 , col = 臂的数量
        #  也可以自行生成每一个实验的mean matrix
        mean_matrix=
        [
        [mean_11,mean_12,mean_13],
        [mean_21,mean_22,mean_23],
        [mean_31,mean_32,mean_33],
        ]

        例如:
            mean_matrix=
        [[  1, 1.1, 1.2],
        [1.4, 1.3,   2],
        [1.1, 1.7, 0.9],
        ]

        """
        mean_matrix_list[i] = mean_matrix 
        n_arms = len(mean_matrix[0])
        n_stages = len(mean_matrix)
        print('There are {} arms with {} stages'.format(n_arms,n_stages))
        print('The number of iterations is {}'.format(iterations ))

        hits_results,traffic_ratio_results,actual_best,regret_list,hits_list = experiment(mean_matrix,decay_list,iterations,algorithms,change_points)

        all_hits_results[i] = hits_results
        all_traffic_ratio_results[i] = traffic_ratio_results
        all_regret_results[i] = regret_list 
        all_hits_list[i]=hits_list
        all_actual_best[i] = actual_best

        #final_data_combine = final_results_to_DataFrame(all_hits_results,all_traffic_ratio_results,all_regret_results,all_hits_list,all_actual_best)

        #final_data_combine.to_csv('final_data_combine_version{}.csv'.format(i))
        print('__________________________________________________________________________________')

