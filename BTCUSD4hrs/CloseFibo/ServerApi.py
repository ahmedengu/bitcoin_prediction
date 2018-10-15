import os
import datetime
import time
from FeatureExtraction import model_weights_path,window_len,dataset_path
from bt_utils import loadData,predict,loadModel,outPutDatabse,outPutMessageFormat
import websockets,asyncio


class ServerApi:
	
	def __init__(self, period, callback,model_weight_path):
		self.model = loadModel(model_weight_path)
		self.time_to_sleep_in_minutes=period*60
		self.callback = callback
		self.today = datetime.datetime.utcnow().date()
		self.model_modified_time = datetime.datetime.fromtimestamp(os.stat(model_weight_path).st_mtime)
		self.dataset_modified_time = datetime.datetime.fromtimestamp(os.stat(dataset_path).st_mtime)
	
	def send_telegram_notification(self, message):
		async def send_message():
			async with websockets.connect('ws://154.16.245.175:8685') as websocket:
				await websocket.send(message)
		asyncio.get_event_loop().run_until_complete(send_message())
	    
	def prediction(self):
		data = loadData(window_len)
		res = predict(self.model,data)
		return res

	
	def run(self):
		while(True):
			if(self.dataset_modified_time != datetime.datetime.fromtimestamp(os.stat(dataset_path).st_mtime)):
				pred=self.prediction()
				outPutDatabse(pred)
				predictionMessage=outPutMessageFormat(pred)
				print(predictionMessage)
				# self.send_telegram_notification(predictionMessage)
				self.dataset_modified_time = datetime.datetime.fromtimestamp(os.stat(dataset_path).st_mtime)
			if(self.model_modified_time!=datetime.datetime.fromtimestamp(os.stat(model_weights_path).st_mtime)):
				self.model.load_weights(model_weights_path)
				self.model_modified_time=datetime.datetime.fromtimestamp(os.stat(model_weights_path).st_mtime)
				print("New Model Loaded ",datetime.datetime.utcnow())
			time.sleep(self.time_to_sleep_in_minutes)
			
if __name__ == "__main__":
    callback = (lambda x: print(x))
    thread = ServerApi(1, callback,model_weights_path)
    thread.run()





