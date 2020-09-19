import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix,classification_report


data = pd.read_csv('College_Data', index_col=0)
print(data.head())
print(data.info())
print(data.describe())

sns.set_style('whitegrid')
sns.lmplot(x='Room.Board',y='Grad.Rate',data=data, hue='Private', fit_reg=False)
plt.show()


sns.lmplot(x='Outstate' ,y='F.Undergrad', data=data, hue='Private', fit_reg=False)
plt.show()


sns.FacetGrid(data=data,hue="Private",palette='coolwarm').map(plt.hist,'Outstate')
plt.show()


sns.FacetGrid(data=data,hue="Private",palette='coolwarm').map(plt.hist,'Grad.Rate')
plt.show()


model = KMeans(n_clusters=2)
model.fit(data.drop('Private',axis=1))

print(model.cluster_centers_)


def getClusterType(cluster):
    return 1 if cluster=='Yes' else 0


data['cluster'] = data['Private'].apply(getClusterType)

print(data.head())

print(confusion_matrix(data['cluster'],model.labels_))
print(classification_report(data['cluster'],model.labels_))