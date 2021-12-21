import pandas as pd
import numpy as np 

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
    version_list= [0.1,0.5,1,5,10] 
    total_backet = 12000
    decay_list = [0.99,0.9,0.8,0.7,0.6,0.5,0.3,0.1] 
    change_points=[0.4,0.5] 
    iterations = 50 

    all_hits_results = dict() 
    all_traffic_ratio_results = dict() 
    all_regret_results=dict() 
    mean_matrix_list = dict()
    all_hits_list=dict() 
    all_actual_best=dict() 

    # There are some algorithms you could choose 
    algorithms = ['TS','DTS','Discounted_epsilon_greedy','epsilon_greedy','softmax_greedy','epsilon_softmax']

    for i in range(len(version_list)):
        print('__________________________start version {}`s training__________________________'.format(i))
        mean_matrix = np.random.rand(3,3) * version_list[i] 
        
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

