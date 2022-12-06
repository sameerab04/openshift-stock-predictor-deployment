
# Get time series data
import yfinance as yf
# Prophet model for time series forecast
#from prophet import Prophet

from prophet import Prophet
# Data processing
#import numpy as np
import pandas as pd
from pandas import DataFrame
from statistics import mean

# Model performance evaluation
from pandas import to_datetime
from matplotlib import pyplot
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/", methods=[ 'POST'])
def get_prediction():
	print("HERE")
	inputs = request.get_json()

	start_date = inputs['start_date']
	end_date = inputs['end_date']
	ticker = inputs['ticker']
	periods = inputs['periods']


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

	# Visualize data using seaborn
# 	sns.set(rc={'figure.figsize':(12,8)})
# 	sns.lineplot(x=data['ds'], y=data['y'])
# 	plt.legend([legend])
# 	plt.show()

	#Create model
	model = Prophet(interval_width=0.99, yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)

	# Fit the model on the training dataset
	#train = data.drop(data.index[-20:])

	train = hist.sample(frac=0.8, random_state=0)
	test_data = data.drop(train.index)
	print(train.tail())

	model.fit(train)
	

	# use the model to make a forecast
	# Make prediction
	future = list()
	future =  pd.date_range(start=start_date, periods = int(periods))
	#future.append([date])
	future = DataFrame(future)
	future.columns = ['ds']
	future['ds'] = to_datetime(future['ds'])

	forecast = model.predict(hist)

	# Merge actual and predicted values
	performance = pd.merge(data, forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds')
	yhat = list(performance['yhat'])
	# Check MAE value
	#performance_MAE = mean_absolute_error(performance['y'], performance['yhat'])
	#print(f'The MAE for the model is {performance_MAE}')

	# Check MAPE value
	y_true, y_pred = pd.array(performance['y']), pd.array(performance['yhat'])
	performance_MAPE = mean(abs((y_true - y_pred) / y_true)) * 100
	
	#print(f'The MAPE for the model is {performance_MAPE}')

	performance['anomaly'] = performance.apply(lambda rows: 1 if ((rows.y<rows.yhat_lower)|(rows.y>rows.yhat_upper)) else 0, axis = 1)
	 # Check the number of anomalies
	performance['anomaly'].value_counts()

	# Take a look at the anomalies
	anomalies = performance[performance['anomaly']==1].sort_values(by='ds')
	num_anomaly = (len(anomalies.index))
	# print(anomalies)
	# print(num_anomaly)
 
	# Visualize the anomalies
	# sns.scatterplot(x='ds', y='y', data=performance, hue='anomaly')
	# sns.lineplot(x='ds', y='yhat', data=performance, color='black')
	#plt.show()
	
	return jsonify(
		number_of_anomalies = num_anomaly,
		model_MAPE = performance_MAPE
	)




if __name__ == "__main__":    
    app.run(host='0.0.0.0')


