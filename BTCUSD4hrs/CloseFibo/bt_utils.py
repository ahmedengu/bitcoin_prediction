#import keras

from keras.models import load_model
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from FeatureExtraction import *
import math
from datetime import date,datetime,timedelta

def loadModel(model_weight_path):
	
	
	bt_model = load_model(model_path)
	
	# load weights
	bt_model.load_weights(model_weight_path)
	print('model loaded sucessfully !!')
	return bt_model


def loadData(window_len):
	
	# url = "https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=" + start_date + "&end=" + end_date
	# bt_market_info = pd.read_html(url)[0]
	
	model_data=extractFeaturesFromData(isTraining=True)
	# need to reverse the data frame so that subsequent rows represent later timepoints
	model_data = model_data.sort_values(by='date')
	
	
	model_data = model_data.drop('date', 1)

	# print('MD ', model_data)
	#val = model_data[-15:]
	data_values = model_data.values
	data_values = data_values.astype('float32')
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaled = scaler.fit_transform(data_values)
	# print('sc shape',scaled.shape)
	scaled=np.expand_dims(scaled,axis=0)
	
	LSTM_testing_inputs = [np.array(LSTM_testing_input) for LSTM_testing_input in scaled]
	LSTM_testing_inputs = np.array(LSTM_testing_inputs)
	LSTM_testing_inputs = [x.ravel() for x in LSTM_testing_inputs]
	LSTM_testing_inputs = np.array(LSTM_testing_inputs)
	# print('shpe',LSTM_testing_inputs.shape)
	# Convert input to 3d vector as lstm accepts 3d vector only
	LSTM_testing_inputs = LSTM_testing_inputs.reshape((LSTM_testing_inputs.shape[0], 1, LSTM_testing_inputs.shape[1]))
	
	return LSTM_testing_inputs


def predict(model, data):
	pred = model.predict(data)
	if pred.argmax(1) == 0:
		res = 'UP_'
		res += str(pred[0][0])
	else:
		res = 'DOWN_'
		res += str(pred[0][1])
	
	return res


def outPutDatabse(pred):
	if (type(pred) == 'list'):
		pred = pred[0]
	pred = pred.split("_")[0]
	bt_market_info = pd.read_csv(dataset_path)
	bt_market_info = bt_market_info.tail(1)
	
	outputframe = pd.DataFrame(data=bt_market_info.close)
	outputframe.index = pd.DatetimeIndex(data=pd.to_datetime(bt_market_info['date']))
	outputframe['instrument'] = "BTSUSD"
	outputframe['forcast'] = pred
	result = pd.read_csv(result_Path)
	if (result.shape[0] > 0):
		result = result.tail(1)
		lastclose = float(result.iloc[:, 1])
		forcast = result.values[:, 3][0]
		close = bt_market_info.values[:, 4][0]
		if close >= lastclose and forcast == "UP" or close <= lastclose and forcast == "DOWN":
			correct, incorrect = 1, 0
		else:
			correct, incorrect = 0, 1
		outputframe['incorrect'] = incorrect
		outputframe['correct'] = correct
		portfolio = math.fabs((close - lastclose) / lastclose) * 100
		portfolio = round(portfolio * 100) / 100
		if (incorrect == 1):
			portfolio *= -1
		outputframe['portfolio'] = portfolio
		outputframe['duration'] = 1
	
	with open(result_Path, 'a') as f:
		outputframe.to_csv(f, header=False)
		
def getLastClose():
	dataset=pd.read_csv(dataset_path)
	dataset=dataset.tail(1)
	return float(dataset.iloc[:,4])
	
def outPutMessageFormat(pred):
	if (type(pred) == 'list'):
		pred = pred[0]
	pred = pred.split("_")[0]
	todaydate = date.today().strftime("%d/%m/%Y")
	timenow = datetime.time(datetime.utcnow()).hour
	forecastedtime = datetime.time(datetime.utcnow() + timedelta(hours=4)).hour
	outputformat = "BTCUSD hourly close\nToday's Date: " + str(todaydate) + "\nToday's Time: " + str(timenow) + \
	               ":00:00\nCurrent BTCUSD Price = $"+str(getLastClose()) +"\nForecasted Time: " + str(forecastedtime) + ":00:00\nPrediction: " + str(pred)
	return outputformat



	