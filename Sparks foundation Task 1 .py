#!/usr/bin/env python
# coding: utf-8

# # THE SPARK FOUNDATION
# TASK 1 - Prediction using Supervised ML.
# 
# To Predict the percentage of marks of the students based on the number of hours they studied
# 
# Author - ANUPAM CHAUHAN

# # Importing libraries

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # Importing Dataset

# In[2]:


data={
    "Hours":[2.5,5.1,3.2,8.5,3.5,1.5,9.2,5.5,8.3,2.7,7.7,5.9,4.5,3.3,1.1,8.9,2.5,1.9,6.1,7.4,2.7,4.8,3.8,6.9,7.8],
    "Scores":[21,47,27,75,30,20,88,60,81,25,85,62,41,42,17,95,30,24,67,69,30,54,35,76,86]
}


# In[3]:


data


# In[4]:


# Converting array to pandas dataframe 
df=pd.DataFrame(data)
df


# # Deciding dependent and independent variables 

# In[5]:


X=df.iloc[:,:-1].values
y=df.iloc[:,1].values


# In[6]:


X


# In[7]:


y


# # Exploratory Data Analysis

# In[8]:


df.plot(x="Hours", y="Scores",style="o",color='hotpink')
plt.title("Hours vs Scores")
plt.xlabel(" Hours")
plt.ylabel("Scores ")
plt.show()


# # Splitting the data into training and testing data 

# In[9]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(X_train,y_train)


# In[10]:


line =model.coef_*X+model.intercept_
plt.scatter(X,y)
plt.plot(X,line);
plt.show();


# In[11]:


y_pred=model.predict(X_test)


# # Comparing actual and predicted score

# In[15]:


df1=pd.DataFrame({'Actaul Score':y_test, 'Predicted Score':y_pred})
df1


# # Predicting Final Score

# In[13]:


hours=np.array(8.6).reshape(-1,1)
own_pred=model.predict(hours)
final_score=str(own_pred)
print(final_score[1:-1])


# # Mean Absolute Error

# In[14]:


from sklearn import metrics
print('Mean Absolute Error:',
     metrics.mean_absolute_error(y_test,y_pred))


# In[ ]:





# In[ ]:





# In[ ]:




