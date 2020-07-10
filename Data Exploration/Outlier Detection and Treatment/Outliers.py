#!/usr/bin/env python
# coding: utf-8

# In[53]:


# import required packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[54]:


# load dataset

df = pd.read_csv("data.csv")


# In[55]:


# print first few rows

df.head()


# # 1. Identification of Outliers

# ## 1.1. Graphical Methods

# ### 1.1.1. Univariate Outlier Detection

# In[56]:


# using boxplot to identify the outliers in the 'Age' variable

df['Age'].plot.box()    # identify outliers in a single variable


# The small circles are the outliers present in the 'Age' variable

# ### 1.1.2. Bivariate Outlier Detection

# In[57]:


# creating scatter plot for 'Age' and 'Fare'

df.plot.scatter('Age','Fare')


# Here we can see that all the values are in the normal range except the top two points

# ## 1.2. Formula Method

# #### 1.2.1. Only for Univariate (Boxplot)
# Outliers lie in: { < Q1 - 1.5 * IQR OR > Q3 + 1.5 * IQR }

# # 2. Treating/Correcting Outliers

# ## 2.1. Remove/Delete the outliers from dataset

# In[58]:


# removing the rows in which there are any outliers present
# remove outliers in 'Fare' variable (from scatter plot we see anything above '300' is outlier for us)

# first fetch the rows for which the values are lesss than '300'

df[df['Fare']<300]    # it will calculate the location of the values for which value of Fare is less than 300 & then subset the entire dataframe using those rows

# df = df[df['Fare']<300]    # this will modify the original dataset by removing the outliers as specified (you can check this by plotting a scatter plot again)


# ## 2.2. Impute outliers like missing values

# In[71]:


# replacing outliers in 'Age' variable with 'mean Age value'
# from the boxplot we can see anything more than 60 to 65 is outlier for us

# locate the values which are outlier values using .loc[] function which takes two arguments, (row,column)

#df.loc[df['Age']>62,'Age'] = df['Age'].mean()
# check the value where the outliers vanishes
df.loc[df['Age']>62,'Age'] = np.mean(df['Age'])    # locate outliers and replace with mean as well as modify the original file


# In[69]:


# check whether we have successfully imputed the outliers with the mean 'Age'

df['Age'].plot.box()


# Here we can see that there are no more outliers present and we have successfully replaced the outliers with the mean age value

# ## 2.3. Transforming & Binning the outliers values
# Ex. 'Log' transforming the original variable

# ## 2.4. Treating the outliers as separately
# Grouping the outliers and non-outliers differently and using different methods/operations for them
