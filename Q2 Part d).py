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

# Lists of the 300 time series, their means and their variances
time_series = []
mean = []
variance = []
for i in range(N):
    mean.append(np.mean(ts.iloc[i]))
    variance.append(np.var(ts.iloc[i]))
    time_series.append(ts.iloc[i])
    
# Returns the mth auto-covariance of a particular time series    
def auto_correlation(m,time_series,mean,variance):
    sigma = 0
    for i in range(len(time_series)-m):
        sigma += (time_series[i]-mean)*(time_series[i+m]-mean)
    return (1/((len(time_series)-m)*variance))*sigma

# x is a list of the 1st order auto_covariances
# y is a list of the 2nd order auto_covariances
x = []
y = []
for i in range(N):
    x.append(auto_correlation(1,time_series[i],mean[i],variance[i]))
    y.append(auto_correlation(2,time_series[i],mean[i],variance[i]))

# Plot
fig, ax = plt.subplots()
scatter = ax.scatter(x,y,s=5,c=cL[0],label = cL[0])
ax.legend(*scatter.legend_elements(),title="Classes")
plt.xlabel("1st order sample auto correlation")
plt.ylabel("2nd order sample auto correlation")

