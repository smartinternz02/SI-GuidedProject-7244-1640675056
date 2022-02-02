#!/usr/bin/env python
# coding: utf-8

# In[26]:


## Importing the libraries
import pandas as pd
import numpy as np
import scipy as sp
import sklearn as sk
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import linear_model

# In[2]:


# Loading the dataset
insurance = pd.read_csv(r"C:\Users\HP\Desktop\insurance.csv")

# In[3]:


### Printing the first five rows
insurance.head()

# In[4]:


### Printing the last five rows
insurance.tail()

# In[5]:


### Printing the summary of the dataframe
insurance.info()

# In[6]:


### Finding the correlation between the columns
insurance.corr()

# In[7]:


### Printing all the column names in the dataset
insurance.columns

# In[28]:


## Checking if there are any NULL values
insurance.isnull().sum()

# In[8]:


### Label Encoding
from sklearn.preprocessing import LabelEncoder

labelencoder_y = LabelEncoder()
insurance['sex'] = labelencoder_y.fit_transform(insurance['sex'])


# In[9]:


def map_smoking(column):
    mapped = []

    for row in column:

        if row == "yes":
            mapped.append(1)
        else:
            mapped.append(0)

    return mapped


insurance["smoker_norm"] = map_smoking(insurance["smoker"])

# In[10]:


nonnum_cols = [col for col in insurance.select_dtypes(include=["object"])]


# In[11]:


def map_obese(column):
    mapped = []
    for row in column:
        if row > 30:
            mapped.append(1)
        else:
            mapped.append(0)
    return mapped


insurance["obese"] = map_obese(insurance["bmi"])

# In[12]:


### Printing the first five rows of the updated dataset
insurance.head()

# In[13]:


### Plotting the graphs using scatter plot
colnum = len(insurance.columns) - 3
fig, ax = plt.subplots(colnum, 1, figsize=(3, 25))
ax[0].set_ylabel("charges")
p_vals = {}
for ind, col in enumerate([i for i in insurance.columns
                           if i not in ["smoker", "region", "expenses", "sex_norm"]]):
    ax[ind].scatter(insurance[col], insurance.expenses, s=5)
    ax[ind].set_xlabel(col)
    ax[ind].set_ylabel("expenses")
plt.show()

# In[14]:


### Dropping the unwanted columns
insurance.drop('bmi', inplace=True, axis=1)
insurance.drop('smoker', inplace=True, axis=1)
insurance.drop('region', inplace=True, axis=1)

# In[15]:


###Printing the updated dataset
print(insurance)

# In[16]:


###Choosing the Dependent and Independent variables
# Independent Variables
x = insurance.iloc[:, [0, 1, 2, 4, 5]]
x

# In[29]:


# Dependent Variable(target column)
y = insurance.iloc[:, 3]
y

# In[18]:


## TRAIN TEST SPLIT

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# In[19]:


## FEATURE SCALING

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()

# In[20]:


x_train

# In[21]:


x_train = sc.fit_transform(x_train)

# In[22]:


x_train

# In[23]:


x_test = sc.transform(x_test)

x_test

# In[24]:


y_test

# In[27]:


## MODEL BUILDING
from sklearn.linear_model import LinearRegression

lr = linear_model.LinearRegression()
lr.fit(x_train, y_train)

# In[30]:


y_pred = lr.predict(x_test)
y_pred

# In[31]:


## Testing for Accuracy
from sklearn.metrics import r2_score

r2_score(y_test, y_pred)

# In[32]:


## Predicting and the result
#lr.predict([[37, 1, 4, 1, 1]])

# Saving our model into a file
import pickle

pickle.dump(lr, open('HIC1.pkl', 'wb'))

# In[ ]:




