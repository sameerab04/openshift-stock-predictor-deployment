
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


def pull_data(start_date, end_date, ticker):

	yfin = yf.Ticker(ticker)

	hist = yfin.history(period="max")

	# Pull close data from Yahoo Finance for the list of tickers
	ticker_list = [ticker]

	data = yf.download(ticker_list, start=start_date, end=end_date)[['Close']]

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
	
	print(data.head(3))
	data.info()
	return data, hist

# Visualize data using seaborn
def visualize_data(data, legend):
	sns.set(rc={'figure.figsize':(12,8)})
	sns.lineplot(x=data['ds'], y=data['y'])
	plt.legend([legend])
	plt.show()

def train_model(data, hist):
	model = Prophet(interval_width=0.99, yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)

	# Fit the model on the training dataset
	#train = data.drop(data.index[-20:])

	train = hist.sample(frac=0.8, random_state=0)
	test_data = data.drop(train.index)
	print(train.tail())

	model.fit(train)
	
	return model, train, test_data

# use the model to make a forecast
# Make prediction
def make_prediction(start_date, end_date, periods, hist,model ):
	future = list()
	future =  pd.date_range(start=start_date, periods = periods)
	#future.append([date])
	future = DataFrame(future)
	future.columns = ['ds']
	future['ds'] = to_datetime(future['ds'])

	forecast = model.predict(hist)
	return forecast

def assess_model(forecast, data):
	# Merge actual and predicted values
	performance = pd.merge(data, forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds')

	# Check MAE value
	performance_MAE = mean_absolute_error(performance['y'], performance['yhat'])
	print(f'The MAE for the model is {performance_MAE}')

	# Check MAPE value
	performance_MAPE = mean_absolute_percentage_error(performance['y'], performance['yhat'])
	print(f'The MAPE for the model is {performance_MAPE}')
	return performance

def create_anomaly_indicator(performance):
	performance['anomaly'] = performance.apply(lambda rows: 1 if ((rows.y<rows.yhat_lower)|(rows.y>rows.yhat_upper)) else 0, axis = 1)
	 # Check the number of anomalies
	performance['anomaly'].value_counts()
	# Take a look at the anomalies
	anomalies = performance[performance['anomaly']==1].sort_values(by='ds')
	print(anomalies)
	return performance
 

def visualize_anomalies(performance):
	# Visualize the anomalies
	sns.scatterplot(x='ds', y='y', data=performance, hue='anomaly')
	sns.lineplot(x='ds', y='yhat', data=performance, color='black')
	plt.show()




start_date = input("Enter start date: ")
end_date = input("Enter end date: ")
ticker = input("Enter ticker: ")
periods = input("Enter number of periods: ")

all_data, history_data = pull_data(start_date, end_date, ticker)
visualize_data(all_data, ticker)
model, train, test_data = train_model(all_data, history_data)
forecast = make_prediction(start_date, end_date, int(periods), history_data, model)
model.plot(forecast); 
plt.show()
performance = assess_model(forecast, all_data)
perfornace_with_anomaly = create_anomaly_indicator(performance)
visualize_anomalies(perfornace_with_anomaly)






