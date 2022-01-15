import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv(r"C:\Users\Owner\Desktop\UQ Year 3 Sem 2 Courses"
                   r"\STAT3006\Assignment 3\Data and Question Sheet\p3_1x.csv")

label = pd.read_csv(r"C:\Users\Owner\Desktop\UQ Year 3 Sem 2 Courses"
                   r"\STAT3006\Assignment 3\Data and Question Sheet\p3_1y.csv")

# Data and labels
x1 = data["xx"]
x2 = data["yy"]
y = label["x"]

# Reformating the labels of the dataframe
for i in range(0,len(data)):
    if y[i] != 1:
        y[i] = -1
        

# Constants
LAMBDA = 1.313131
N = len(data)
A = np.identity(3) 
A[0][0] = 0


# Calculating THETA
v = np.array([y,x1*y,x2*y]).T
v_sum = 0
v_matrix = 0

for i in range(0,N):
    v_sum += v[i].reshape(1,3)
    v_matrix += v[i].reshape(3,1)@v[i].reshape(1,3)
    
THETA = (np.linalg.inv(LAMBDA*N*A + v_matrix))@v_sum.T


# Returns 1 if classification is correct, 0 if incorrect
def classification(x1,x2,y,Lambda):    
    p = float(np.array([1,x1,x2])@(np.linalg.inv(Lambda*N*A + v_matrix))@v_sum.T)
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
        c.append(classification(x1[i],x2[i],y[i],Lambda))
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
    

# Drawing the decision boundary based on the optimal lambda
plt.figure()
plt.scatter(x1,x2,s=10,c=y)
X1 = np.linspace(-1,1,100)
X2 = (-THETA[0] - (THETA[1])*X1)/THETA[2]
plt.ylim((-1.5,0.5))
plt.plot(X1,X2)
plt.xlabel('xx')
plt.ylabel('yy')
plt.title('Plot of the decision boundary using the optimal lambda')

print("Chosen lambda: ", optimal_lambda)