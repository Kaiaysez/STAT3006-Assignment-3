import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\Owner\Desktop\UQ Year 3 Sem 2 Courses"
                   r"\STAT3006\Assignment 3\Data and Question Sheet\data_banknote_authentication.txt", header=None)

# Transform the data
# Uncomment for part e) and f)
#df.iloc[:,1] = df.iloc[:,1].transform(lambda x: x**3) #0.00801 lambda 7.474747


# Reformating the labels of the dataframe
for i in range(0,len(df)):
    if df[4][i] != 1:
        df[4][i] = -1
        
# Constants
# Note that this LAMBDA is for testing purposes only
# The functions below require a lambda as input
LAMBDA = 2.121212 
N = len(df)
A = np.identity(5) 
A[0][0] = 0

 
# Data and labels
x1 = np.array(df[0])
x2 = np.array(df[1])
x3 = np.array(df[2])
x4 = np.array(df[3])
y = np.array(df[4])


# Calculating THETA
v = np.array([y,x1*y,x2*y,x3*y,x4*y]).T
v_sum = 0
v_matrix = 0
for i in range(0,N):
    v_sum += v[i].reshape(1,5)
    v_matrix += v[i].reshape(5,1)@v[i].reshape(1,5)
THETA = (np.linalg.inv(LAMBDA*N*A + v_matrix))@v_sum.T


# Returns 1 if classification is correct, 0 if incorrect
def classification(x1,x2,x3,x4,y,Lambda):    
    p = float(np.array([1,x1,x2,x3,x4])@(np.linalg.inv(Lambda*N*A + v_matrix))@v_sum.T)
    if p == 0:
        p = -1
    if p*y > 0:
        return 1
    else:
        return 0
    
# List of classification results: 1 if correct, 0 if incorrect
def results(Lambda):
    c = list()
    for i in range(0,N):
        c.append(classification(x1[i],x2[i],x3[i],x4[i],y[i],Lambda))
    return c
    

# Estimate of the in sample empirical classification risk
def sample_empirical_risk(Lambda):
    misclassified = 0
    for i in results(Lambda):
        if i == 0:
            misclassified += 1
    return misclassified/N

# Find the optimal lambda by minimizing the sample empirical risk 
risk = []
X1 = np.linspace(0,10,100)
for i in range(len(X1)):
    risk.append(sample_empirical_risk(X1[i]))
plt.figure()
plt.plot(X1,risk)
plt.xlabel('Lambda')
plt.ylabel('Sample empirical risk')
plt.title('Sample empirical risk vs lambda')
optimal_lambda = X1[np.argmin(risk)]

    
# Pairplot of the data (forged banknotes vs real banknotes)
f = sns.pairplot(df, hue = 4, size = 5)
f.map_offdiag(sns.kdeplot, levels=4, color=".2")

# Pairplot of the data (correctly classified vs incorrectly classified)
df2 = df.copy()
del df2[4]
df2["c"] = results(LAMBDA)
g = sns.pairplot(df2, hue = "c", size = 5)
g.map_offdiag(sns.kdeplot, levels=4, color=".2")

print("Chosen lambda: ",optimal_lambda)
print("Sample empirical risk: ",sample_empirical_risk(optimal_lambda))