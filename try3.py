
import cv2
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import kNN
from numpy import *
import operator
# executing whole codes with sample data [18, 90]
global labels
labels = kNN.createDataSet()

age = input("Enter age group : ")
diseases =input("Enter Disease : ")
num =input("Enter Disease Num: ")

if diseases == 'Diabetes' :
     group = array([[30, 2.0], [40, 6.7], [50, 15.1], [60, 19.6], [70,27.9],
                   [30, 11.3], [40, 19.2], [50, 32.1], [60, 46.9], [70, 64.7],
                   [30, 11.5], [40, 18.1], [50, 28.2], [60, 33.7], [70, 35.4],
                   [30, 14.9], [40, 19.8], [50, 16.3], [60, 15.6], [70, 11.9],
                   [30, 0.2], [40, 0.4], [50, 3.1], [60, 5.4], [70,16.6]])
elif diseases == 'Hypertenstion':
    group = array([[30, 2.0], [40, 6.7], [50, 15.1], [60, 19.6], [70,27.9],
                   [30, 11.5], [40, 18.1], [50, 28.2], [60, 33.7], [70, 35.4],
                   [30, 14.9], [40, 19.8], [50, 16.3], [60, 15.6], [70, 11.9],
                   [30, 0.2], [40, 0.4], [50, 3.1], [60, 5.4], [70,16.6]])
    labels.remove(diseases)
elif diseases == 'Hypercholesterolemia':
    group = array([[30, 2.0], [40, 6.7], [50, 15.1], [60, 19.6], [70,27.9],
                   [30, 11.3], [40, 19.2], [50, 32.1], [60, 46.9], [70, 64.7],
                   [30, 14.9], [40, 19.8], [50, 16.3], [60, 15.6], [70, 11.9],
                   [30, 0.2], [40, 0.4], [50, 3.1], [60, 5.4], [70,16.6]])
elif diseases == 'Hypertriglyceridemia':
    group = array([[30, 2.0], [40, 6.7], [50, 15.1], [60, 19.6], [70,27.9],
                   [30, 11.3], [40, 19.2], [50, 32.1], [60, 46.9], [70, 64.7],
                   [30, 11.5], [40, 18.1], [50, 28.2], [60, 33.7], [70, 35.4],
                   [30, 0.2], [40, 0.4], [50, 3.1], [60, 5.4], [70,16.6]])
elif diseases == 'Chronic-Kidney-Diseases':
     group = array([[30, 2.0], [40, 6.7], [50, 15.1], [60, 19.6], [70,27.9],
                   [30, 11.3], [40, 19.2], [50, 32.1], [60, 46.9], [70, 64.7],
                   [30, 11.5], [40, 18.1], [50, 28.2], [60, 33.7], [70, 35.4],
                   [30, 14.9], [40, 19.8], [50, 16.3], [60, 15.6], [70, 11.9]])
while diseases in labels :
          labels.remove(diseases)
print(kNN.classify([int(age), int(num)], group, labels, 3))

df = pd.read_excel('diabetes_highblood.xlsx')
df['new'] = np.nan
df.loc[(int(age)-30)/10,['new']] = int(num)
df.head()
print(df)
sns.regplot(x=df["age"], y=df["diabetes"], fit_reg=False)
sns.regplot(x=df["age"], y=df["hypertension"], color='red', fit_reg = False)
sns.regplot(x=df["age"], y=df["hypercholesterolemia"], color='purple', fit_reg = False)
sns.regplot(x=df["age"], y=df["hypertriglyceridemia"], color='orange', fit_reg = False)
sns.regplot(x=df["age"], y=df["percent"], color='yellow', fit_reg = False)
ax = sns.regplot(x=df["age"], y=df["new"], color='green', fit_reg = False, scatter_kws={'s': 200})
ax.set_xlabel('Age')
ax.set_ylabel('Prevalence')
plt.show()

