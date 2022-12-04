#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Get time series data
import yfinance as yf
# Prophet model for time series forecast
from prophet import Prophet
# Data processing
import numpy as np
import pandas as pd
from pandas import DataFrame
# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
# Model performance evaluation
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from pandas import to_datetime
from matplotlib import pyplot


# In[3]:


# Data start date
start_date = '2020-01-02'
# Data end date. yfinance excludes the end date, so we need to add one day to the last day of data
end_date = '2022-11-20'

yfin = yf.Ticker('GOOG')

hist = yfin.history(period="max")

# Pull close data from Yahoo Finance for the list of tickers
ticker_list = ['GOOG']

#data = yf.download(ticker_list, start=start_date, end=end_date)[['Close']]

hist = yfin.history(period="max")
hist = hist[['Close']]
hist.index = hist.index.tz_convert(None)
hist = hist.dropna()
hist.reset_index(level=0, inplace=True)
# Change column names
hist = hist.rename({'Date': 'ds', 'Close': 'y'}, axis='columns')
# Change column names
data = hist.copy()

#data = data.reset_index()
#data.columns = ['ds', 'y']
# Take a look at the data
data.head()


# In[4]:


# Information on the dataframe
data.info()


# In[5]:


# Visualize data using seaborn
sns.set(rc={'figure.figsize':(12,8)})
sns.lineplot(x=data['ds'], y=data['y'])
plt.legend(['Google'])


# In[7]:


# Add seasonality
model = Prophet(interval_width=0.99, yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)


# In[8]:


# Fit the model on the training dataset
#train = data.drop(data.index[-20:])

train = hist.sample(frac=0.8, random_state=0)
test_data = data.drop(train.index)
train.tail()


# In[9]:


model.fit(train)


# In[15]:


# use the model to make a forecast
# Make prediction

future = list()
date1 = "2022-10-02"  #  start date
date2 = "2022-11-14"  #  end date
future =  pd.date_range(start=date1, periods = 365)
#future.append([date])
future = DataFrame(future)
future.columns = ['ds']
future['ds'] = to_datetime(future['ds'])

forecast = model.predict(hist)
forecast


# In[16]:


# Visualize the forecast
model.plot(forecast); # Add semi-colon to remove the duplicated chart


# In[17]:


# Merge actual and predicted values
performance = pd.merge(data, forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds')


# In[19]:


# Check MAE value
performance_MAE = mean_absolute_error(performance['y'], performance['yhat'])
print(f'The MAE for the model is {performance_MAE}')


# In[20]:


# Check MAPE value
performance_MAPE = mean_absolute_percentage_error(performance['y'], performance['yhat'])
print(f'The MAPE for the model is {performance_MAPE}')


# In[21]:


# Create an anomaly indicator
performance['anomaly'] = performance.apply(lambda rows: 1 if ((rows.y<rows.yhat_lower)|(rows.y>rows.yhat_upper)) else 0, axis = 1)


# In[22]:


# Check the number of anomalies
performance['anomaly'].value_counts()


# In[23]:


# Take a look at the anomalies
anomalies = performance[performance['anomaly']==1].sort_values(by='ds')
anomalies


# In[24]:


# Visualize the anomalies
sns.scatterplot(x='ds', y='y', data=performance, hue='anomaly')
sns.lineplot(x='ds', y='yhat', data=performance, color='black')


# In[ ]:




