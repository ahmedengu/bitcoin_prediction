
# import the relevant Keras modules
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM,GRU
from keras.layers import Dropout
from keras import metrics
import keras.backend as K
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class ModelUtils(object):
    '''
    classdocs
    '''

    def __init__(self, params):
        '''
        Constructor
        '''
        
    @staticmethod    
    def r2_keras(y_true, y_pred):
        """
        Coefficient of Determination
        """    
        SS_res = K.sum(K.square(y_true - y_pred))
        SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
        return (1 - SS_res / (SS_tot + K.epsilon()))


    @staticmethod
    def buildLstmInput(dataset, norm_cols, window_len):
        """
        Model output is next price normalised to 10th previous closing price
        Return array of sequences
        """
        # array of sequences    
        LSTM_inputs = []
        for i in range(len(dataset) - window_len):
            # Get a windows of rows
            print (i)
            temp_set = dataset[i:(i + window_len)].copy()
            temp_set2 = dataset[i+1:(i + window_len)].copy()
            # Normalize from -1 to 1

            for col in norm_cols:
                #temp_set.loc[:, col] = temp_set[col] / temp_set[col].iloc[0] - 1


                A=temp_set.loc[:,col][1:]
                B=temp_set[col][:-1]
                C=[]
                for i,j in zip(A,B):
                    C.append(float(i)/(j+0.0000001)-1)
                temp_set2.loc[:,col].iloc[:] = C


            LSTM_inputs.append(temp_set2)

        #pca = PCA(n_components=5)
        #scaler = StandardScaler()
        #scaler.fit(LSTM_inputs[0])
        #pca.fit(scaler.transform(LSTM_inputs[0]))
        #pca.fit((LSTM_inputs[0]))

        #X_sc = [ scaler.transform((x)) for x in LSTM_inputs]
        #X_pca_train = pca.fit_transform(LSTM_inputs)

        return LSTM_inputs
    
    @staticmethod
    def buildLstmOutput(dataset, target, window_len):
        """
        Model output is next price normalised to 10th previous closing price
        """    
        res= (dataset[target][window_len:].values / dataset[target][window_len-1:-1].values) - 1
        #res= (dataset[target][window_len:].values / dataset[target][:-window_len].values) - 1
        #res = dataset[target][window_len:].values
        res2=[]
        for i in range(len(res)):
            if res[i]>0:
                res2.append([1,0])
            else:
                res2.append([0, 1])

        return res2

    @staticmethod
    def build_model(inputs, output_size, neurons_lv1, neurons_lv2,neurons_lv3,neurons_lv4, activ_func="tanh",
                    dropout=0.3, loss="mean_squared_error", optimizer="adam"):
        """
        LSTM
        """
        model = Sequential()
        
        #model.add(LSTM( input_shape=(inputs.shape[1], inputs.shape[2]),units=neurons_lv1,return_sequences=False,activation=activ_func))
        
        #model.add(Dropout(dropout))
        
        #model.add(LSTM( units=neurons_lv2, return_sequences=False,recurrent_dropout=0.1))

        #model.add(Dropout(dropout))

        model.add(Dense(input_dim=len(inputs[0]),units=neurons_lv1,activation=activ_func))
        model.add(Dropout(dropout))
        model.add(Dense(units=neurons_lv2, activation=activ_func))
        model.add(Dropout(dropout))
        model.add(Dense(units=neurons_lv3, activation=activ_func))
        model.add(Dropout(dropout))
        
        
        
        model.add(Dense(units=output_size))
        
        model.add(Activation("softmax"))
        
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy', metrics.mse])
        # Summarize model
        model.summary()
        return model
