#!/usr/bin/env python
# coding: utf-8

# Decision Tree
#  
# Assignment
# 
# 
# About the data: 
# Let’s consider a Company dataset with around 10 variables and 400 records. 
# The attributes are as follows: 
#  Sales -- Unit sales (in thousands) at each location
# 
#  Competitor Price -- Price charged by competitor at each location
# 
#  Income -- Community income level (in thousands of dollars)
# 
#  Advertising -- Local advertising budget for company at each location (in thousands of dollars)
# 
#  Population -- Population size in region (in thousands)
# 
#  Price -- Price company charges for car seats at each site
# 
#  Shelf Location at stores -- A factor with levels Bad, Good and Medium indicating the quality of the shelving location for the car seats at each site
# 
#  Age -- Average age of the local population
# 
#  Education -- Education level at each location
# 
#  Urban -- A factor with levels No and Yes to indicate whether the store is in an urban or rural location
# 
#  US -- A factor with levels No and Yes to indicate whether the store is in the US or not
# 
# The company dataset looks like this:
# 
# Problem Statement:
# A cloth manufacturing company is interested to know about the segment or attributes causes high sale. 
# Approach - A decision tree can be built with target variable Sale (we will first convert it in categorical variable) & all other variable will be independent in the analysis.  

# In[1]:


get_ipython().system('pip install category_encoders')
import category_encoders as ce
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn import preprocessing


# In[2]:


company = pd.read_csv(r"C:\Users\anupa\Downloads\Company_Data.csv")


# In[3]:


company.head(10)


# In[4]:


company.shape


# In[5]:


company.info


# In[6]:


company.describe()


# In[7]:


company.isnull().sum()


# In[8]:


import category_encoders as ce
# encode variables with ordinal encoding
encoder = ce.OrdinalEncoder(cols=['ShelveLoc', 'Urban', 'US'])
company1 = encoder.fit_transform(company)


# In[9]:


company_val = []
for value in company["Sales"]:
     if value<=7.49:
        company_val.append("low")
     else:
        company_val.append("high")

company1["company_val"]= company_val


# In[10]:


company1.head(10)


# In[11]:


x = company1.drop(['company_val','Sales'], axis =1)
y = company1['company_val']


# In[12]:


x


# In[13]:


y


# In[14]:


x_train, x_test,y_train,y_test = train_test_split(x,y, test_size=0.2,random_state=40)


# In[15]:


model1 = DecisionTreeClassifier(criterion = 'entropy',max_depth=4)
model1.fit(x_train,y_train)


# In[16]:


#Predicting on test data
pred_test1 = model1.predict(x_test)
#Accuracy on test data
print('Test data Accuracy is:',np.mean(pred_test1==y_test)*100)
#Predicting on train data
pred_train1 = model1.predict(x_train)
#Accuracy on train data
print('Train data Accuracy is:',np.mean(pred_train1==y_train)*100)


# In[17]:


model2 = DecisionTreeClassifier(criterion = 'entropy',max_depth=6)
model2.fit(x_train,y_train)


# In[18]:


#Predicting on test data
pred_test2 = model2.predict(x_test)
#Accuracy on test data
print('Test data Accuracy is:',np.mean(pred_test2==y_test)*100)
#Predicting on train data
pred_train2 = model2.predict(x_train)
#Accuracy on train data
print('Train data Accuracy is:',np.mean(pred_train2==y_train)*100)


# In[19]:


model3 = DecisionTreeClassifier(criterion = 'entropy',max_depth=7)
model3.fit(x_train,y_train)


# In[20]:


#Predicting on test data
pred_test3 = model3.predict(x_test)
#Accuracy on test data
print('Test data Accuracy is:',np.mean(pred_test3==y_test))
#Predicting on train data
pred_train3 = model3.predict(x_train)
#Accuracy on train data
print('Train data Accuracy is:',np.mean(pred_train3==y_train))


# In[21]:


fig = plt.figure(figsize=(25,20))
fig = tree.plot_tree(model2,
 feature_names= ['CompPrice','Income','Advertising','Population','Price', 'ShelveLoc', 'Age', 'Education',
 'Urban', 'US'], class_names= ['low', 'high'], filled=True)
plt.title('Decision tree using Entropy',fontsize=22)
plt.savefig('DT_Entropy.pdf')


# In[22]:


from sklearn.tree import DecisionTreeClassifier
model_gini1 = DecisionTreeClassifier(criterion='gini', max_depth=5)
model_gini1.fit(x_train,y_train)


# In[23]:


#Predicting on test data
pred_testg1 = model_gini1.predict(x_test)
#Accuracy on test data
print('Test data Accuracy is:',np.mean(pred_testg1==y_test)*100)
#Predicting on train data
pred_traing1 = model_gini1.predict(x_train)
#Accuracy on train data
print('Train data Accuracy is:',np.mean(pred_traing1==y_train)*100)


# In[24]:


model_gini2 = DecisionTreeClassifier(criterion='gini', max_depth=6)
model_gini2.fit(x_train,y_train)


# In[25]:


#Predicting on test data
pred_testg2 = model_gini2.predict(x_test)
#Accuracy on test data
print('Test data Accuracy is:',np.mean(pred_testg2==y_test)*100)
#Predicting on train data
pred_traing2 = model_gini2.predict(x_train)
#Accuracy on train data
print('Train data Accuracy is:',np.mean(pred_traing2==y_train)*100)


# In[26]:


model_gini3 = DecisionTreeClassifier(criterion='gini', max_depth=7)
model_gini3.fit(x_train,y_train)


# In[27]:


#Predicting on test data
pred_testg3 = model_gini3.predict(x_test)
#Accuracy on test data
print('Test data Accuracy is:',np.mean(pred_testg3==y_test)*100)
#Predicting on train data
pred_traing3 = model_gini3.predict(x_train)
#Accuracy on train data
print('Train data Accuracy is:',np.mean(pred_traing3==y_train)*100)


# In[28]:


# let's plot the decision tree
fig = plt.figure(figsize=(30,25))
fig = tree.plot_tree(model_gini1,
 feature_names= ['CompPrice','Income','Advertising','Population','Price', 'ShelveLoc', 'Age', 'Education',
 'Urban', 'US'], class_names= ['low', 'high'], filled=True)
plt.title('Decision tree using Gini',fontsize=22)
plt.savefig('DT_Gini.pdf')


# In[29]:


preds = model_gini3.predict(x_test)
pd.Series(preds).value_counts()


# In[30]:


preds


# In[31]:


pd.crosstab(y_test,preds)


# In[32]:


np.mean(preds==y_test)*100


# In[33]:


import seaborn as sns
from sklearn.metrics import confusion_matrix
sns.heatmap(confusion_matrix(y_test, preds),annot=True,fmt = "d")
print('Classification Report ',classification_report(y_test,preds))


# In[ ]:




