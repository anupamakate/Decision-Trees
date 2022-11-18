#!/usr/bin/env python
# coding: utf-8

# Use decision trees to prepare a model on fraud data 
# treating those who have taxable_income <= 30000 as "Risky" and others are "Good"
# 
# Data Description :
# 
# Undergrad : person is under graduated or not
# Marital.Status : marital status of a person
# Taxable.Income : Taxable income is the amount of how much tax an individual owes to the government 
# Work Experience : Work experience of an individual person
# Urban : Whether that person belongs to urban area or not
# 
# 

# In[1]:


#importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets  
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import  DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn import preprocessing

df = pd.read_csv(r"C:\Users\anupa\Downloads\Fraud_check.csv")


# In[2]:


#Viewing top 5 rows of dataframe
df.head()


# In[3]:


df.tail()


# In[4]:


#Creating dummy vairables for ['Undergrad','Marital.Status','Urban'] dropping first dummy variable
df=pd.get_dummies(df,columns=['Undergrad','Marital.Status','Urban'], drop_first=True)


# In[5]:


#Creating new cols TaxInc and dividing 'Taxable.Income' cols on the basis of [10002,30000,99620] for Risky and Good
df["TaxInc"] = pd.cut(df["Taxable.Income"], bins = [10002,30000,99620], labels = ["Risky", "Good"])


# In[6]:


print(df)


# # Lets assume: taxable_income <= 30000 as “Risky=0” and others are “Good=1”

# In[7]:


#After creation of new col. TaxInc also made its dummies var concating right side of df
df = pd.get_dummies(df,columns = ["TaxInc"],drop_first=True)


# In[8]:


#Viewing buttom 10 observations
df.tail(10)


# In[9]:


# let's plot pair plot to visualise the attributes all at once
import seaborn as sns
sns.pairplot(data=df, hue = 'TaxInc_Good')


# In[10]:


# Normalization function 
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)


# In[11]:


# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(df.iloc[:,1:])
df_norm.tail(10)


# In[12]:


# Declaring features & target
X = df_norm.drop(['TaxInc_Good'], axis=1)
y = df_norm['TaxInc_Good']


# In[13]:


from sklearn.model_selection import train_test_split


# In[14]:


# Splitting data into train & test
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=0)


# In[15]:


##Converting the Taxable income variable to bucketing. 
df_norm["income"]="<=30000"
df_norm.loc[df["Taxable.Income"]>=30000,"income"]="Good"
df_norm.loc[df["Taxable.Income"]<=30000,"income"]="Risky"


# In[16]:


##Droping the Taxable income variable
df.drop(["Taxable.Income"],axis=1,inplace=True)


# In[17]:


df.rename(columns={"Undergrad":"undergrad","Marital.Status":"marital","City.Population":"population","Work.Experience":"experience","Urban":"urban"},inplace=True)
## As we are getting error as "ValueError: could not convert string to float: 'YES'".
## Model.fit doesnt not consider String. So, we encode


# In[18]:


from sklearn import preprocessing
le=preprocessing.LabelEncoder()
for column_name in df.columns:
    if df[column_name].dtype == object:
        df[column_name] = le.fit_transform(df[column_name])
    else:
        pass


# In[19]:


##Splitting the data into featuers and labels
features = df.iloc[:,0:5]
labels = df.iloc[:,5]


# In[20]:


## Collecting the column names
colnames = list(df.columns)
predictors = colnames[0:5]
target = colnames[5]
##Splitting the data into train and test


# In[21]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(features,labels,test_size = 0.2,stratify = labels)


# In[22]:


##Model building
from sklearn.ensemble import RandomForestClassifier as RF
model = RF(n_jobs = 3,n_estimators = 15, oob_score = True, criterion = "entropy")
model.fit(x_train,y_train)


# In[23]:


model.estimators_
model.classes_
model.n_features_
model.n_classes_


# In[24]:


model.n_outputs_


# In[25]:


model.oob_score_
###74.7833%


# In[26]:


##Predictions on train data
prediction = model.predict(x_train)


# In[27]:


##Accuracy
# For accuracy 
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_train,prediction)
##98.33%


# In[28]:


np.mean(prediction == y_train)
##98.33%


# In[29]:


##Confusion matrix
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_train,prediction)


# In[30]:


##Prediction on test data
pred_test = model.predict(x_test)


# In[31]:


##Accuracy
acc_test =accuracy_score(y_test,pred_test)
##78.333%


# In[32]:


pip install pydotplus


# In[33]:


conda install -c anaconda pydotplus


# In[37]:


## In random forest we can plot a Decision tree present in Random forest
from sklearn.tree import export_graphviz
import pydotplus
from six import StringIO


# In[38]:


tree = model.estimators_[5]


# In[42]:


dot_data = StringIO()
export_graphviz(tree,out_file = dot_data, filled = True,rounded = True, feature_names = predictors ,class_names = target,impurity =False) 


# In[43]:


graph = pydotplus.graph_from_dot_data(dot_data.getvalue())


# # Building Decision Tree Classifier using Entropy Criteria

# In[44]:


model = DecisionTreeClassifier(criterion = 'entropy',max_depth=3)
model.fit(x_train,y_train)


# In[45]:


from sklearn import tree


# In[46]:


#PLot the decision tree
tree.plot_tree(model);


# In[47]:


colnames = list(df.columns)
colnames


# In[48]:


fn=['population','experience','Undergrad_YES','Marital.Status_Married','Marital.Status_Single','Urban_YES']
cn=['1', '0']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(model,
               feature_names = fn, 
               class_names=cn,
               filled = True);


# In[49]:


#Predicting on test data
preds = model.predict(x_test) # predicting on test data set 
pd.Series(preds).value_counts() # getting the count of each category 


# In[50]:


preds


# In[51]:


pd.crosstab(y_test,preds) # getting the 2 way table to understand the correct and wrong predictions


# In[52]:


# Accuracy 
np.mean(preds==y_test)


# # Building Decision Tree Classifier (CART) using Gini Criteria

# In[53]:


from sklearn.tree import DecisionTreeClassifier
model_gini = DecisionTreeClassifier(criterion='gini', max_depth=3)


# In[54]:


model_gini.fit(x_train, y_train)


# In[55]:


#Prediction and computing the accuracy
pred=model.predict(x_test)
np.mean(preds==y_test)


# # Decision Tree Regression Example

# In[56]:


# Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor


# In[57]:


array = df.values
X = array[:,0:3]
y = array[:,3]


# In[58]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)


# In[59]:


model = DecisionTreeRegressor()
model.fit(X_train, y_train)


# In[60]:


#Find the accuracy
model.score(X_test,y_test)


# In[ ]:




