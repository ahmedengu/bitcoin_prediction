# import the relevant Keras modules
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM,GRU
from keras.layers import Dropout
from keras import metrics
import keras.backend as K
from keras.regularizers import L1L2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from keras import regularizers

class ModelUtils(object):
    def __init__(self, params):
        '''
        Constructor
        '''
        
    @staticmethod
    def buildLstmInput(dataset, norm_cols, window_len):
        '''
        Create window and normalize the values in the range (-1, 1)
        '''
        normalized_values=[]
        for i in range(len(dataset)-window_len):
            val=dataset[i:i+window_len]
            data_values=val.values
            data_values = data_values.astype('float32')
            # normalize features
            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaled = scaler.fit_transform(data_values)
            normalized_values.append(scaled)
        return normalized_values
    


    
    @staticmethod
    def buildLstmOutput(dataset, target, window_len):
        '''
        Output will be the signal up or down
        '''
        output_values= (dataset[target][window_len:].values / dataset[target][window_len-1:-1].values) - 1
        output_class=[]
        for i in range(len(output_values)):
            if output_values[i]>0:
                output_class.append([1,0])
            else:
                output_class.append([0, 1])

        return output_class


    @staticmethod
    def build_model(inputsShape, output_size, neurons_lv1, neurons_lv2,neurons_lv3,neurons_lv4):#op size is 2
        """
        Build LSTM
        """
        
        model = Sequential()
        model.add(LSTM(neurons_lv1, input_shape=(inputsShape[0], inputsShape[1]),return_sequences=True,recurrent_dropout=.3,kernel_regularizer=regularizers.l2(0.01)))#input_shape=(inputs.shape[1], inputs.shape[2])
        model.add(LSTM(neurons_lv2,activation="relu",return_sequences=False,recurrent_dropout=.3))
        model.add(Dense(neurons_lv3))
        model.add(Dropout(0.3))
        model.add(Dense(neurons_lv3,activation="relu"))
        model.add(Dropout(0.3))
        model.add(Dense(output_size,activation="sigmoid"))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
        model.summary()
        return model

