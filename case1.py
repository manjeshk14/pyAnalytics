#python:Topic

#importing a csv filr
url = 'https://raw.githubusercontent.com/DUanalytics/datasets/master/csv/denco.csv'
import pandas as pd
df= pd.read_csv(url)
df #check if data is imported
 #DIMENSIONS OF DATA
df.shape
df.head(3)
import numpy as np
df.head()
df.describe()
df.columns
df.dtypes
pd.options.display.float_format='{:.2f}'.format # to see approx values
df['region']=df['region'].astype('category') # column type has changed to category
#quick summary
df.region.value_counts()
df.region.value_counts().plot(kind='bar')
df.max(axis=0) #customer with maximum 
df.dtypes
df.custname.value_counts().sort_values(ascending=False).head(5)
df.custname.value_counts().sort_values(ascending=False).tail(5)
df.custname.value_counts().sort_values(ascending=True).head(5)

#revenue total per customer
df.groupby('custname').revenue.sum()
df.groupby('custname').revenue.sum().sort_values(ascending=False).head(5)
df.groupby('custname').aggregate({'revenue':[np.sum,max,min,'size']})
df.groupby('custname').aggregate({'revenue':[np.sum,max,min,'size']})

#q3
df[('partnum')['revenue']].sort_values(ascending=False).head(5)
