import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import fmin_tnc
import numpy as np
import bisect 

def PowerLawDegree(max_degree,min_degree,exponent):
    Degree_Dist = np.zeros([int(max_degree - min_degree) + 1,2])
    for index,value in enumerate(range(int(min_degree),int(max_degree) + 1)):
        Degree_Dist[index,0] = value
        Degree_Dist[index,1] = value**(-exponent)

    #normalize distribution to sum to 1
    Degree_Dist[:,1] = Degree_Dist[:,1]/np.sum(Degree_Dist[:,1])
    Mean_Degree = np.dot(Degree_Dist[:,1].transpose(),Degree_Dist[:,0])
    Square_Degree = np.multiply(Degree_Dist[:,0],Degree_Dist[:,0])
    Second_Moment = np.dot(Degree_Dist[:,1].transpose(),Square_Degree)
    #print 'Mean Degree:', Mean_Degree
    #print 'Second Moment', Second_Moment
    return Mean_Degree,Second_Moment,Degree_Dist




#evaluate differences between proposed distribution and desired distribution
def DDistMatch(x, min_degree = 4, mean = 8, nodes = 4000, second_moment = 12*8):
    #change second moment to be a global variable
    max_degree = np.sqrt(nodes*mean)
    PL_Mean,PL_2nd_Mnt,D = PowerLawDegree(max_degree,min_degree,x[0])
    Poiss_Mean = min_degree + x[1]
    Poiss_2nd_Mnt = x[1]**2 + (2*min_degree + 1)*x[1] + min_degree**2
    Total_Mean = np.dot([Poiss_Mean,PL_Mean],[x[2],1-x[2]])
    Total_2nd_Mnt = np.dot([Poiss_2nd_Mnt,PL_2nd_Mnt],[x[2],1-x[2]])
    Current_Stats = np.array([Total_Mean,Total_2nd_Mnt])
    Desired_Stats = np.array([mean,second_moment])
    Current_Stats
    Error = np.linalg.norm(Current_Stats - Desired_Stats)

    return Error

def MCMC_Degree_Seq(nodes, ave, max_degree_count, second_moment, trials = True, d = True, min_degree = 5, threshold = .1, allowed_error = .025):
    #threshold, if proposed updated degree sequence is worse in matching prescribed moments, likelihood of accepting it
    max_degree = np.floor(np.sqrt(nodes*ave + 1))

    if trials:
        trials = 20*ave*nodes

    if d:
        d = ave*np.ones(nodes)

    current_second_moment = np.dot(d,d)/(1.0*np.sum(d))

    for index in range(max_degree_count):
        d[index] = max_degree

    for trial in range(trials):
        Success = False
        #print trial
        while not Success:
            Percent_Error = np.abs(current_second_moment - second_moment)/second_moment

            index_plus = np.random.randint(nodes)
            index_minus = np.random.randint(nodes)

            #need to choose degree to change based on whether that increases or decreases
            #with a certain probability threshold.  If need to increase, 75% choose larger.
            #If need to decrease 25% choose smaller.  f(distance_from_goal, parameter)

            proposed_second_moment = current_second_moment + (2.0*d[index_plus] - 2.0*d[index_minus])/np.sum(d)

            Proposed_Percent_Error = np.abs(proposed_second_moment - second_moment)/second_moment

            if Proposed_Percent_Error > Percent_Error and Proposed_Percent_Error > allowed_error:
                delta = np.abs(d[index_plus] - d[index_minus])
                NewThreshold = 2*threshold/delta
                if np.random.binomial(1,NewThreshold) == 0:
                    index_plus,index_minus = index_minus, index_plus
                    proposed_second_moment = current_second_moment + (2.0*d[index_plus] - 2.0*d[index_minus])/np.sum(d)

            if d[index_plus] < max_degree:
                if d[index_minus] > min_degree:
                    Success = True
                    d[index_plus] += 1
                    d[index_minus] -= 1
                    current_second_moment = proposed_second_moment
                    print(current_second_moment, d[index_plus], d[index_minus])
                    print('')
    return d

def Collect_Degree_Seq(degree_seq_count,nodes,ave,max_degree_count, second_moment):
    d_list = []
    Threshold = .1
    Trials = 2*ave*nodes
    for count in range(degree_seq_count):


        d = MCMC_Degree_Seq(nodes, ave, max_degree_count, second_moment, trials = Trials, threshold = Threshold)

        Second_Moment_Actual = np.dot(d,d)/np.sum(d)
        Delta = np.abs(Second_Moment_Actual - second_moment)/second_moment

        #Threshold = .95*np.floor(np.sqrt(nodes*ave))
        #Max_Node_Count = np.sum(d >= Threshold)


        if Delta < .05:# and Max_Node_Count >= 5:
            d_list.append(d)
            #print count, 'Success', Delta, Trials, Threshold

        else:
            Threshold = .9*Threshold
            Trials = 1.1*Trials
            #print count, 'Failure', Delta, Trials, Threshold

    return d_list

#just do negative binomial across each row
#keep track of number of flips available S_*
#cum sum of degree sequence [1,2,3]  --> [1,3,6,15,21]
#-use bisection method to find next hit.  O(NlnN)  get stop marks,

def Chung_Lu_Approx(d):
    #fast approximation of Chung Lu algorithm
    G = np.zeros([len(d),len(d)])
    d_sum = np.sum(d)
    d_cum_sum = np.hstack([0,np.cumsum(d)])
    for i in range(len(d)):
        current_index = i+1
        while current_index <= len(d):
            next_edge = np.random.negative_binomial(1,d[i]/d_sum)
            current_index = find_next_index(d_cum_sum,current_index,next_edge)
            if current_index <= len(d):
                G[current_index-1,i],G[i,current_index-1] = 1,1
            current_index += 1
    return G

def find_next_index(d,current_index,next_edge):
    start_count = d[current_index]
    #find ? such that d[?] <= next_edge + start_count < d[?+1],  return ?
    #https://docs.python.org/3/library/bisect.html
    answer = bisect.bisect_right(d,next_edge+start_count)
    return answer



def Chung_Lu_RG(d):
    G = np.zeros([len(d), len(d)])
    for i in range(len(d)):
        print(i)
        for j in range(len(d - i)):
            if i != j:
                p = np.min([d[i]*d[j]/(1.0*np.sum(d)),1])
                G[i,j] = np.random.binomial(1,p)
                G[j,i] = G[i,j]
    return G
