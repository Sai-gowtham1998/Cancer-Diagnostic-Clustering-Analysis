#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df = pd.read_csv(r'C:\Gowtham\Finger tips\All Projects\Python + ML\ML Project - Clustering Cancer Analysis\cancer_diagnostic_features.csv')


# In[4]:


df.isnull().sum()


# # Printing information about dataset

# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


df.head()


# In[7]:


df_1 = df.drop('UnderRisk',axis=1)


# # Kmean clustering on dataset 

# In[8]:


from sklearn.cluster import KMeans


# In[9]:


cls = KMeans(n_clusters = 2, )
cls.fit(df_1)


# # Checking the wcss score

# In[9]:


wcss = cls.inertia_
wcss


# # Trying different n and find wcss score

# In[10]:


#create empty list
wcss = []
#select k value from 1 to 10
for i in range(1, 11):
    cls = KMeans(n_clusters = i, random_state = 42)
    cls.fit(df_1)
    # inertia method returns wcss for that model
    wcss.append(cls.inertia_)


# # Plot all wcss score

# In[11]:


plt.figure(figsize=(10,5))
sns.lineplot(range(1, 11), wcss,marker='o',color='red')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# # Try again kmeans with best no. cluster according wo wcss score

# In[17]:


# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 6, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(df_1)



# In[18]:


y_kmeans


# # Print cluster centers 

# In[19]:


kmeans.cluster_centers_


# # Create column cluster for predicted labels value

# In[20]:


df_1['cluster']=  y_kmeans
df_1.head()


# # Plot the hierarchical clustering using scipy 

# In[12]:


df_h = df.drop('UnderRisk',axis=1)


# In[13]:


#The following linkage methods are used to compute the distance between two clusters 
# method='ward' uses the Ward variance minimization algorithm
from scipy.cluster.hierarchy import linkage,dendrogram
merg = linkage(df_1, method = "ward")
#Plot the hierarchical clustering as a dendrogram.
#leaf_rotation : double, optional Specifies the angle (in degrees) to rotate the leaf labels.

dendrogram(merg, leaf_rotation = 90)
plt.xlabel("data points")
plt.ylabel("euclidean distance")
plt.show()


# # Applying AgglomerativeClustering using number of cluster

# In[16]:


from sklearn.cluster import AgglomerativeClustering


# In[17]:


hc = AgglomerativeClustering(n_clusters = 6, affinity = "euclidean", linkage = "ward")
cluster = hc.fit_predict(df_h)


# # Create label column for predicted cluster label

# In[18]:


df_h["label"] = cluster


# In[19]:


df_h.head()


# # Show label counts 

# In[27]:


df_h.label.value_counts()


# # Show a silhouette score

# In[28]:


from sklearn.metrics import silhouette_score


# In[29]:


score_agg = silhouette_score(df_h, cluster)
score_agg


# In[ ]:




