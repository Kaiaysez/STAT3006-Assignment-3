import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ts = pd.read_csv(r"C:\Users\Owner\Desktop\UQ Year 3 Sem 2 Courses"
                   r"\STAT3006\Assignment 3\Data and Question Sheet\p2_1ts.csv")
cL = pd.read_csv(r"C:\Users\Owner\Desktop\UQ Year 3 Sem 2 Courses"
                   r"\STAT3006\Assignment 3\Data and Question Sheet\p2_1cL.csv")

# Getting rid of the extra column
del ts[ts.columns[0]]
del cL[cL.columns[0]]

# Renaming columns as integers starting from 0
ts.columns = range(ts.shape[1])
cL.columns = range(cL.shape[1])

# Constants
N = len(ts)

# Class proportions
prop_y1 = 0
prop_y2 = 0
prop_y3 = 0
for i in range(N):
    if int(np.array(cL.iloc[i])) == 1:
        prop_y1 += 1
    elif int(np.array(cL.iloc[i])) == 2:
        prop_y2 += 1
    else:
        prop_y3 += 1
prop_y1 = prop_y1/N
prop_y2 = prop_y2/N
prop_y3 = prop_y3/N

# Beta's
B_num_1 = 0
B_num_2 = 0
B_num_3 = 0
B_denom_1 = 0
B_denom_2 = 0
B_denom_3 = 0
for i in range(N):
    if int(np.array(cL.iloc[i])) == 1:
        for r in range(1,100):
            B_num_1 += ts.iloc[i][r]*ts.iloc[i][r-1]
            B_denom_1 += ts.iloc[i][r-1]**2
    elif int(np.array(cL.iloc[i])) == 2:
        for r in range(1,100):
            B_num_2 += ts.iloc[i][r]*ts.iloc[i][r-1]
            B_denom_2 += ts.iloc[i][r-1]**2
    else:
        for r in range(1,100):
            B_num_3 += ts.iloc[i][r]*ts.iloc[i][r-1]    
            B_denom_3 += ts.iloc[i][r-1]**2 
Beta_1 = B_num_1/B_denom_1      
Beta_2 = B_num_2/B_denom_2  
Beta_3 = B_num_3/B_denom_3

# Class variances
s2_num_1 = 0   
s2_num_2 = 0   
s2_num_3 = 0   

for i in range(N):
    if int(np.array(cL.iloc[i])) == 1:
        s2_num_1 += ts.iloc[i][0]**2
        for r in range(1,100):
            s2_num_1 += (ts.iloc[i][r]-Beta_1*ts.iloc[i][r-1])**2
    elif int(np.array(cL.iloc[i])) == 2:
        s2_num_2 += ts.iloc[i][0]**2
        for r in range(1,100):
            s2_num_2 += (ts.iloc[i][r]-Beta_2*ts.iloc[i][r-1])**2
    else:
        s2_num_3 += ts.iloc[i][0]**2
        for r in range(1,100):
            s2_num_3 += (ts.iloc[i][r]-Beta_3*ts.iloc[i][r-1])**2
           
s2_1 = s2_num_1/(prop_y1*100*N)
s2_2 = s2_num_2/(prop_y2*100*N)
s2_3 = s2_num_3/(prop_y3*100*N)     
        

# Implementing the Bayes classifier 
# Only use the numerator of tau_k since tau_k has the same denominator for all k

def tau_1(x):
    summation = 0
    for r in range(1,100):
        summation += (x[r]-Beta_1*x[r-1])**2
    return(((1/(2*np.pi*s2_1))**50)*prop_y1*np.e**((-1/(2*s2_1))*((x[0])**2+summation)))

def tau_2(x):
    summation = 0
    for r in range(1,100):
        summation += (x[r]-Beta_2*x[r-1])**2
    return(((1/(2*np.pi*s2_2))**50)*prop_y2*np.e**((-1/(2*s2_2))*((x[0])**2+summation)))
    
def tau_3(x):
    summation = 0
    for r in range(1,100):
        summation += (x[r]-Beta_3*x[r-1])**2
    return(((1/(2*np.pi*s2_3))**50)*prop_y3*np.e**((-1/(2*s2_3))*((x[0])**2+summation)))
        
        
        
# MLE
MLE = [prop_y1,prop_y2,prop_y3,Beta_1,Beta_2,Beta_3,s2_1,s2_2,s2_3]

# List containing predicted (by the Bayes classifier) classes of each observation     
classification = []
for i in range(N):
    classification.append(np.argmax(np.array([tau_1(np.array(ts.iloc[i])),tau_2(np.array(ts.iloc[i])),tau_3(np.array(ts.iloc[i]))])))       
        
empirical_risk = 0
for i in range(N):
    if (classification[i]+1) != int(cL.iloc[i]):
        empirical_risk += 1
empirical_risk = empirical_risk/N

print("MLE vector: ", MLE)
print("Empirical risk is ", empirical_risk)