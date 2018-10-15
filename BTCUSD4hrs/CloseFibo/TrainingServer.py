
from ModelUtils import ModelUtils

from sklearn.model_selection import train_test_split
from keras.layers import *
from matplotlib import pyplot
from FeatureExtraction import *
from sklearn import metrics
from datetime import datetime
import time
from FeatureExtraction import dataset_path


def trainTheModel():


    model_data=extractFeaturesFromData()
    print("Training Start")
    #need to reverse the data frame so that subsequent rows represent later timepoints
    model_data = model_data.sort_values(by='date')
    
    
    # #create Training and Test set
    # training_set, test_set = model_data[model_data['date'] < split_date], model_data[model_data['date'] >= split_date]##2 yr data for training and 4 mnth for testing
    # test_set=test_set[test_set['date']<=end_date]

    # training_set,test_set=train_test_split(model_data,shuffle=False,test_size=.3)
    training_set=model_data

    # # we don't need the date columns anymore
    training_set = training_set.drop('date', 1)
    
    # test_set = test_set.drop('date', 1)
    
    #Prepare model input and output
    LSTM_training_inputs = ModelUtils.buildLstmInput(training_set, None, window_len)
    LSTM_training_outputs = ModelUtils.buildLstmOutput(training_set, 'bt_close', window_len)
    
    # LSTM_test_inputs = ModelUtils.buildLstmInput(test_set, None, window_len)
    # LSTM_test_outputs = ModelUtils.buildLstmOutput(test_set, 'bt_close', window_len)

    print("\nNumber Of Input Training's sequences: {}".format(len(LSTM_training_inputs)))
    print("\nNumber Of Output Training's sequences: {}".format(len(LSTM_training_outputs)))
    # print("\nNumber Of Input Test's sequences: {}".format(len(LSTM_test_inputs)))
    # print("\nNumber Of Output Test's sequences: {}".format(len(LSTM_test_outputs)))
    
    #convert to numpy array
    LSTM_training_inputs = [np.array(LSTM_training_input) for LSTM_training_input in LSTM_training_inputs]
    LSTM_training_inputs = np.array(LSTM_training_inputs)
    LSTM_training_inputs=[x.ravel() for x in LSTM_training_inputs]
    LSTM_training_inputs = np.array(LSTM_training_inputs)
    
    LSTM_training_outputs = [np.array(LSTM_training_output) for LSTM_training_output in LSTM_training_outputs]
    LSTM_training_outputs = np.array(LSTM_training_outputs)
    LSTM_training_outputs=[x.ravel() for x in LSTM_training_outputs]
    LSTM_training_outputs = np.array(LSTM_training_outputs)
    
    # LSTM_test_inputs = [np.array(LSTM_test_input) for LSTM_test_input in LSTM_test_inputs]
    # LSTM_test_inputs = np.array(LSTM_test_inputs)
    # LSTM_test_inputs = [x.ravel() for x in LSTM_test_inputs]
    # LSTM_test_inputs = np.array(LSTM_test_inputs)
    #
    # LSTM_test_outputs = [np.array(LSTM_test_output) for LSTM_test_output in LSTM_test_outputs]
    # LSTM_test_outputs = np.array(LSTM_test_outputs)
    # LSTM_test_outputs = [x.ravel() for x in LSTM_test_outputs]
    # LSTM_test_outputs = np.array(LSTM_test_outputs)

    #reshape input to be 3D [samples, timesteps, features] as LSTM accepts only 3D vector
    train_X = LSTM_training_inputs.reshape((LSTM_training_inputs.shape[0], 1, LSTM_training_inputs.shape[1]))
    # test_X = LSTM_test_inputs.reshape((LSTM_test_inputs.shape[0], 1, LSTM_test_inputs.shape[1]))
    # print(train_X.shape, LSTM_training_outputs.shape, test_X.shape, LSTM_test_outputs.shape)
    
    #Build the model
    model = ModelUtils.build_model(train_X.shape[1:], output_size=2, neurons_lv1=num_of_neurons_lv1, neurons_lv2=num_of_neurons_lv2,
                                          neurons_lv3=num_of_neurons_lv3,neurons_lv4=num_of_neurons_lv4)
    # #Fit data to the model
    # history = model.fit(train_X, LSTM_training_outputs, epochs=200, batch_size=3000, validation_data=(test_X,LSTM_test_outputs), verbose=2, shuffle=False)
    history = model.fit(train_X, LSTM_training_outputs, epochs=300, batch_size=3000, verbose=2, shuffle=False)

    # #Make predictions to find accuracy
    # pre = (model.predict(test_X))
    #
    # #Classification report
    # b = np.zeros_like(pre)
    # b[np.arange(len(pre)), pre.argmax(1)] = 1
    # print( metrics.classification_report(LSTM_test_outputs,b))
    
    #Save trained model
    
    model.save(model_path)
    model.save_weights(model_weights_path)
    print('model saved')


datenow=lambda :datetime.utcnow().day
timetospeep = 20 * 60
DategaptoTrain = timeToTrainInDays
nextTrainingDate=datenow()
lastTraingDate=datenow()-1
while True:
    if (datenow() ==nextTrainingDate and lastTraingDate!=datenow()):
        print("trianing started at ", datetime.utcnow())
        startTime=datetime.utcnow()
        trainTheModel()
        lastTraingDate=datenow()
        nextTrainingDate=datenow()+DategaptoTrain
        print("trianing Ended at ", datetime.utcnow(),' in time ',datetime.utcnow()-startTime)
    time.sleep(timetospeep)