#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 02:56:03 2018

@author: amr
"""

'''
Created on 24 nov 2017

@author: mantica

'''
import matplotlib

matplotlib.use('Agg')

import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image
import io
import urllib

#import keras
from PlotUtils import PlotUtils
from ModelUtils import ModelUtils
from keras.optimizers import Adam,SGD
from keras.callbacks import ModelCheckpoint
from datetime import datetime,timedelta

pridection_date = '20180316'

#Moving Average
def MA(df, n):
    MA = pd.Series(df['bt_Close'][::-1].rolling(n).mean(), name = 'bt_MA_' + str(n))[::-1]
    df = df.join(MA.fillna(0))
    return df

#Exponential Moving Average
def EMA(df, n):
    EMA = pd.Series(df['bt_Close'][::-1].ewm(span = n, min_periods = n - 1).mean(), name = 'bt_EMA_' + str(n))[::-1]
    df = df.join(EMA.fillna(0))
    return df

#Momentum
def MOM(df, n):
    M = pd.Series(df['bt_Close'][::-1].diff(n), name = 'bt_MOM_' + str(n))[::-1]
    df = df.join(M.fillna(0))
    return df

#Rate of Change
def ROC(df, n):
    M = df['bt_Close'][::-1].diff(n - 1)
    N = df['bt_Close'][::-1].shift(n - 1)
    ROC = pd.Series(M / N, name = 'bt_ROC_' + str(n))[::-1]
    df = df.join(ROC.fillna(0))
    return df

#Average True Range  
def ATR(df, n):  
    i = 0  
    TR_l = [0]  
    while i < df.index[-1]:  
        TR = max(df.get_value(i + 1, 'bt_High'), df.get_value(i, 'bt_Close'))\
        - min(df.get_value(i + 1, 'bt_Low'), df.get_value(i, 'bt_Close'))  
        TR_l.append(TR)  
        i = i + 1  
    TR_s = pd.Series(TR_l)  
    ATR = pd.Series(TR_s[::-1].ewm( span = n, min_periods = n).mean(),\
                    name = 'bt_ATR_' + str(n))[::-1]  
    df = df.join(ATR.fillna(0))  
    return df

#Bollinger Bands  
def BBANDS(df, n):  
    MA = pd.Series(df['bt_Close'][::-1].rolling(n).mean()) [::-1] 
    MSD = pd.Series(df['bt_Close'][::-1].rolling(n).std())  [::-1]
    b1 = 4 * MSD / MA  
    B1 = pd.Series(b1[::-1], name = 'bt_BollingerB1_' + str(n))  [::-1]
    df = df.join(B1.fillna(0))  
    b2 = (df['bt_Close'][::-1] - MA + 2 * MSD) / (4 * MSD)  [::-1]
    B2 = pd.Series(b2[::-1], name = 'bt_BollingerB2_' + str(n))  [::-1]
    df = df.join(B2.fillna(0))  
    return df

#Pivot Points, Supports and Resistances  
def PPSR(df):  
    PP = pd.Series((df['bt_High'] + df['bt_Low'] + df['bt_Close']) / 3)  
    R1 = pd.Series(2 * PP - df['bt_Low'])  
    S1 = pd.Series(2 * PP - df['bt_High'])  
    R2 = pd.Series(PP + df['bt_High'] - df['bt_Low'])  
    S2 = pd.Series(PP - df['bt_High'] + df['bt_Low'])  
    R3 = pd.Series(df['bt_High'] + 2 * (PP - df['bt_Low']))  
    S3 = pd.Series(df['bt_Low'] - 2 * (df['bt_High'] - PP))  
    psr = {'bt_PP':PP, 'bt_R1':R1, 'bt_S1':S1, 'bt_R2':R2,\
           'bt_S2':S2, 'bt_R3':R3, 'bt_S3':S3}  
    PSR = pd.DataFrame(psr)  
    df = df.join(PSR.fillna(0))  
    return df

#Stochastic oscillator %K  
def STOK(df):  
    SOk = pd.Series((df['bt_Close'] - df['bt_Low']) / \
                    (df['bt_High'] - df['bt_Low']), name = 'bt_SOK')  
    df = df.join(SOk.fillna(0))  
    return df

# Stochastic Oscillator, EMA smoothing, nS = slowing (1 if no slowing)  
def STO(df,  nK, nD, nS=1):  
    SOk = pd.Series((df['bt_Close'][::-1] - df['bt_Low'][::-1].rolling(nK).min()) / (df['bt_High'][::-1].rolling(nK).max() - df['bt_Low'][::-1].rolling(nK).min()), name = 'bt_SOK_'+str(nK)) [::-1] 
    SOd = pd.Series(SOk[::-1].ewm(ignore_na=False, span=nD, min_periods=nD-1, adjust=True).mean(), name = 'bt_SOD_'+str(nD))  [::-1]
    SOk = SOk.ewm(ignore_na=False, span=nS, min_periods=nS-1, adjust=True).mean()  
    SOd = SOd.ewm(ignore_na=False, span=nS, min_periods=nS-1, adjust=True).mean()   
    df = df.join(SOk.fillna(0))
    df = df.join(SOd.fillna(0))  
    return df  

#Trix  
def TRIX(df, n):  
    EX1 = df['bt_Close'][::-1].ewm(span = n, min_periods = n - 1).mean()[::-1]
    EX2 = EX1[::-1].ewm(span = n, min_periods = n - 1).mean()[::-1]
    EX3 = EX2[::-1].ewm(span = n, min_periods = n - 1).mean() [::-1]
    i = 0  
    ROC_l = [0]  
    while i + 1 <= df.index[-1]:  
        ROC = (EX3[i + 1] - EX3[i]) / EX3[i]  
        ROC_l.append(ROC)  
        i = i + 1  
    Trix = pd.Series(ROC_l, name = 'bt_Trix_' + str(n))  
    df = df.join(Trix.fillna(0))  
    return df

#Average Directional Movement Index  
def ADX(df, n, n_ADX):  
    i = 0  
    UpI = []  
    DoI = []  
    while i + 1 <= df.index[-1]:  
        UpMove = df.get_value(i + 1, 'bt_High') - df.get_value(i, 'bt_High')  
        DoMove = df.get_value(i, 'bt_Low') - df.get_value(i + 1, 'bt_Low')  
        if UpMove > DoMove and UpMove > 0:  
            UpD = UpMove  
        else: UpD = 0  
        UpI.append(UpD)  
        if DoMove > UpMove and DoMove > 0:  
            DoD = DoMove  
        else: DoD = 0  
        DoI.append(DoD)  
        i = i + 1  
    i = 0  
    TR_l = [0]  
    while i < df.index[-1]:  
        TR = max(df.get_value(i + 1, 'bt_High'), df.get_value(i, 'bt_Close')) - min(df.get_value(i + 1, 'bt_Low'), df.get_value(i, 'bt_Close'))  
        TR_l.append(TR)  
        i = i + 1  
    TR_s = pd.Series(TR_l)  
    ATR = pd.Series(TR_s[::-1].ewm( span = n, min_periods = n).mean())  [::-1]
    UpI = pd.Series(UpI)  
    DoI = pd.Series(DoI)  
    PosDI = pd.Series(UpI[::-1].ewm( span = n, min_periods = n - 1).mean()[::-1] / (ATR)) 
    NegDI = pd.Series(DoI[::-1].ewm( span = n, min_periods = n - 1).mean()[::-1] / (ATR))  
    ADX = pd.Series((abs(PosDI - NegDI) / (PosDI + NegDI))[::-1].ewm( span = n_ADX, min_periods = n_ADX - 1).mean(), name = 'bt_ADX_' + str(n))  [::-1]
    df = df.join(ADX.fillna(0))  
    return df

#MACD, MACD Signal and MACD difference  
def MACD(df, n_fast, n_slow):  
    EMAfast = pd.Series(df['bt_Close'][::-1].ewm( span = n_fast, min_periods = n_slow - 1).mean()) [::-1] 
    EMAslow = pd.Series(df['bt_Close'][::-1].ewm( span = n_slow, min_periods = n_slow - 1).mean()) [::-1] 
    MACD = pd.Series(EMAfast - EMAslow, name = 'bt_MACD_' + str(n_fast) + '_' + str(n_slow))  
    MACDsign = pd.Series(MACD[::-1].ewm( span = 9, min_periods = 8).mean(), name = 'bt_MACDsign_' + str(n_fast) + '_' + str(n_slow))  [::-1]
    MACDdiff = pd.Series(MACD - MACDsign, name = 'bt_MACDdiff_' + str(n_fast) + '_' + str(n_slow))  
    df = df.join(MACD.fillna(0))  
    df = df.join(MACDsign.fillna(0))  
    df = df.join(MACDdiff.fillna(0))  
    return df

#Mass Index  
def MassI(df):  
    Range = df['bt_High'] - df['bt_Low']  
    EX1 = Range[::-1].ewm( span = 9, min_periods = 8).mean() [::-1]
    EX2 = EX1[::-1].ewm( span = 9, min_periods = 8).mean()[::-1]
    Mass = EX1 / EX2  
    MassI = pd.Series(Mass[::-1].rolling( 25).sum(), name = 'bt_Mass_Index')  [::-1]
    df = df.join(MassI.fillna(0))  
    return df

#Vortex Indicator: http://www.vortexindicator.com/VFX_VORTEX.PDF  
def Vortex(df, n):  
    i = 0  
    TR = [0]  
    while i < df.index[-1]:  
        Range = max(df.get_value(i + 1, 'bt_High'), df.get_value(i, 'bt_Close')) - min(df.get_value(i + 1, 'bt_Low'), df.get_value(i, 'bt_Close'))  
        TR.append(Range)  
        i = i + 1  
    i = 0  
    VM = [0]  
    while i < df.index[-1]:  
        Range = abs(df.get_value(i + 1, 'bt_High') - df.get_value(i, 'bt_Low')) - abs(df.get_value(i + 1, 'bt_Low') - df.get_value(i, 'bt_High'))  
        VM.append(Range)  
        i = i + 1  
    VI = pd.Series(pd.Series(VM)[::-1].rolling( n).sum() / pd.Series(TR)[::-1].rolling( n).sum(), name = 'bt_Vortex_' + str(n))  [::-1]
    df = df.join(VI.fillna(0))  
    return df

#KST Oscillator  
def KST(df): 
    r1=10
    r2=15
    r3=20
    r4=30
    n1=10
    n2=10
    n3=10
    n4=15
    M = df['bt_Close'][::-1].diff(r1 - 1)[::-1]  
    N = df['bt_Close'][::-1].shift(r1 - 1)  [::-1]
    ROC1 = M / N  
    M = df['bt_Close'][::-1].diff(r2 - 1) [::-1] 
    N = df['bt_Close'][::-1].shift(r2 - 1)[::-1]  
    ROC2 = M / N  
    M = df['bt_Close'][::-1].diff(r3 - 1) [::-1] 
    N = df['bt_Close'][::-1].shift(r3 - 1)[::-1]  
    ROC3 = M / N  
    M = df['bt_Close'][::-1].diff(r4 - 1) [::-1] 
    N = df['bt_Close'][::-1].shift(r4 - 1)[::-1]  
    ROC4 = M / N  
    KST = pd.Series(ROC1[::-1].rolling( n1).sum() + ROC2[::-1].rolling( n2).sum() * 2 + ROC3[::-1].rolling( n3).sum() * 3 + ROC4[::-1].rolling( n4).sum() * 4, name = 'bt_KST')[::-1]  
    df = df.join(KST.fillna(0))  
    return df


#Relative Strength Index  
def RSI(df, n):  
    i = 0  
    UpI = [0]  
    DoI = [0]  
    while i + 1 <= df.index[-1]:  
        UpMove = df.get_value(i + 1, 'bt_High') - df.get_value(i, 'bt_High')  
        DoMove = df.get_value(i, 'bt_Low') - df.get_value(i + 1, 'bt_Low')  
        if UpMove > DoMove and UpMove > 0:  
            UpD = UpMove  
        else: UpD = 0  
        UpI.append(UpD)  
        if DoMove > UpMove and DoMove > 0:  
            DoD = DoMove  
        else: DoD = 0  
        DoI.append(DoD)  
        i = i + 1  
    UpI = pd.Series(UpI)  
    DoI = pd.Series(DoI)  
    PosDI = pd.Series(UpI[::-1].ewm( span = n, min_periods = n - 1).mean()) [::-1] 
    NegDI = pd.Series(DoI[::-1].ewm( span = n, min_periods = n - 1).mean())  [::-1]
    RSI = pd.Series(PosDI / (PosDI + NegDI), name = 'bt_RSI_' + str(n))  
    df = df.join(RSI.fillna(0))  
    return df

#True Strength Index  
def TSI(df, r, s):  
    M = pd.Series(df['bt_Close'].diff(1))  
    aM = abs(M)  
    EMA1 = pd.Series(M[::-1].ewm( span = r, min_periods = r - 1).mean())  [::-1]
    aEMA1 = pd.Series(aM[::-1].ewm( span = r, min_periods = r - 1).mean())  [::-1]
    EMA2 = pd.Series(EMA1[::-1].ewm( span = s, min_periods = s - 1).mean()) [::-1] 
    aEMA2 = pd.Series(aEMA1[::-1].ewm( span = s, min_periods = s - 1).mean()) [::-1]  
    TSI = pd.Series(EMA2 / aEMA2, name = 'bt_TSI_' + str(r) + '_' + str(s))  
    df = df.join(TSI.fillna(0))  
    return df

if __name__ == '__main__':   
    
    ########################
    # Configuration
    ######################## 
    # random seed for reproducibility
    # np.random.seed(202)
    model_path = "weights_85%Train_15%Test.hdf5"
    
    date_format = datetime.strptime(pridection_date, '%Y%m%d')
    date_format=date_format-timedelta(days=50)
    start_date = date_format.strftime('%Y%m%d')
    
    date_format = datetime.strptime(pridection_date, '%Y%m%d')
    date_format=date_format-timedelta(days=1)
    end_date = date_format.strftime('%Y%m%d')
    
    date_format = datetime.strptime(pridection_date, '%Y%m%d')
    date_format=date_format-timedelta(days=5)
    split_date = date_format.strftime('%Y-%m-%d')
     

    # Our LSTM model will use previous data to predict the next day's closing price of bitcoin. 
    # We must decide how many previous days it will have access to
    window_len = 5
    bt_epochs = 2500
    bt_batch_size = 8
    num_of_neurons_lv1 = 128
    num_of_neurons_lv2 = 64
    num_of_neurons_lv3 = 4
    num_of_neurons_lv4 = 16
    activ_func="tanh"
    dropout=0.8
    loss="categorical_crossentropy"
    optimizer=Adam(lr=0.0003)

    
    ###################################
    # Getting the BT
    ###################################
    
    bt_img = urllib.request.urlopen("http://logok.org/wp-content/uploads/2016/10/Bitcoin-Logo-640x480.png")
    image_file = io.BytesIO(bt_img.read())
    bt_im = Image.open(image_file)
    width_bt_im , height_bt_im = bt_im.size
    bt_im = bt_im.resize((int(bt_im.size[0] * 0.8), int(bt_im.size[1] * 0.8)), Image.ANTIALIAS)
    
    ################
    # Data Ingestion
    ################
    
    # get market info for bitcoin from the start of 2016 to the current day
    url="https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=" + start_date + "&end=" + end_date
    bt_market_info = pd.read_html(url)[0]
    # convert the date string to the correct date format
    bt_market_info = bt_market_info.assign(Date=pd.to_datetime(bt_market_info['Date']))
    # look at the first few rows
    bt_market_info = bt_market_info.rename(columns={"Close**":"Close","Open*":"Open"})
    print("BT")
    print(bt_market_info.head())
    print('\nshape: {}'.format(bt_market_info.shape)) 
    print("\n")
    #PlotUtils.plotCoinTrend(bt_market_info, bt_im, "Bitcoin")
    
    # Feature Eng
    print("Feature ENG")
    bt_market_info.columns = [bt_market_info.columns[0]] + ['bt_' + i for i in bt_market_info.columns[1:]]
    for coins in ['bt_']: 
        kwargs = { coins + 'day_diff': lambda x: (x[coins + 'Close'] - x[coins + 'Open']) / x[coins + 'Open']}
        bt_market_info = bt_market_info.assign(**kwargs)
    print('\nshape: {}'.format(bt_market_info.shape))
    print(bt_market_info.head())
    print("\n")
    #PlotUtils.plotCoinTrainingTest(bt_market_info, split_date, bt_im,"bt_Close","Bitcoin")
    
    ###########################################################################################################
    # DATA PREPARATION
    # In time series models, we generally train on one period of time and then test on another separate period.
    # I've created a new data frame called model_data. 
    # I've removed some of the previous columns (open price, daily highs and lows) and reformulated some new ones.
    # close_off_high represents the gap between the closing price and price high for that day, where values of -1 and 1 
    # mean the closing price was equal to the daily low or daily high, respectively. 
    # The volatility columns are simply the difference between high and low price divided by the opening price.
    # You may also notice that model_data is arranged in order of earliest to latest. 
    # We don't actually need the date column anymore, as that information won't be fed into the model.
    ###########################################################################################################
    # close_off_high = 2 * (High - Close) / (High - Low) - 1
    # volatility = (High - Low) / Open

    for coins in ['bt_']:
        kwargs = { coins + 'close_off_high': lambda x: 2 * (x[coins + 'High'] - x[coins + 'Close']) / (x[coins + 'High'] - x[coins + 'Low']) - 1,
                coins + 'volatility': lambda x: (x[coins + 'High'] - x[coins + 'Low']) / (x[coins + 'Open'])}
        bt_market_info = bt_market_info.assign(**kwargs)

    bt_market_info=MA(bt_market_info,window_len)
    bt_market_info = EMA(bt_market_info, window_len)
    bt_market_info = MOM(bt_market_info, window_len)
    bt_market_info = ROC(bt_market_info, window_len)
    bt_market_info = ATR(bt_market_info, window_len)
    bt_market_info = BBANDS(bt_market_info, window_len)
    bt_market_info = PPSR(bt_market_info)
    bt_market_info = STOK(bt_market_info)
    bt_market_info = STO(bt_market_info,window_len,window_len)
    bt_market_info = TRIX(bt_market_info,window_len)
    bt_market_info = ADX(bt_market_info,window_len,window_len*2)
    bt_market_info = MACD(bt_market_info,window_len*2,window_len)
    bt_market_info = MassI(bt_market_info)
    bt_market_info = Vortex(bt_market_info,window_len)
    bt_market_info = KST(bt_market_info)
    bt_market_info = RSI(bt_market_info,window_len)
    bt_market_info = TSI(bt_market_info,window_len*2,window_len)

    model_data = bt_market_info[['Date'] + [coin + metric for coin in [ 'bt_'] \
            for metric in ['Close',  'close_off_high', 'volatility', 'day_diff','High','Low','Open','Volume',\
                           'Market Cap','MA_5','EMA_5','MOM_5','ROC_5','ATR_5','BollingerB1_5','BollingerB2_5',\
                           'PP','R1','S1','R2','S2','R3','S3','SOK','SOK_5','SOD_5','Trix_5','ADX_5',\
                           'MACD_10_5','MACDsign_10_5','MACDdiff_10_5','Mass_Index','Vortex_5','KST' \
                           ,'RSI_5','TSI_10_5']]]
    #,'Volume','Market Cap'
    
    # need to reverse the data frame so that subsequent rows represent later timepoints
    model_data = model_data.sort_values(by='Date')
    print("Model Data")
    print('\nshape: {}'.format(model_data.shape))
    print(model_data.head())
    print("\n")
    
    # create Training and Test set    
    test_set = model_data[model_data['Date'] >= split_date]
    
    new_data = pd.DataFrame(test_set[-1:].values, columns=test_set.columns)
    test_set = test_set.append(new_data)

    # we don't need the date columns anymore
    test_set = test_set.drop('Date', 1)
    
    norm_cols = [coin + metric for coin in ['bt_'] for metric in \
                 ['Close', 'High', 'Low', 'Open','Volume','Market Cap',\
                  'MA_5','EMA_5','MOM_5','ROC_5','ATR_5','BollingerB1_5','BollingerB2_5',\
                  'PP','R1','S1','R2','S2','R3','S3','SOK','SOK_5','SOD_5','Trix_5','ADX_5',\
                  'MACD_10_5','MACDsign_10_5','MACDdiff_10_5','Mass_Index','Vortex_5','KST' \
                  ,'RSI_5','TSI_10_5']]
    #
    
    
    LSTM_test_inputs = ModelUtils.buildLstmInput(test_set, norm_cols, window_len)
    # model output is next price normalised to 10th previous closing price
    LSTM_test_outputs = ModelUtils.buildLstmOutput(test_set, 'bt_Close', window_len)
    
    
    print("\nNumber Of Input Test's sequences: {}".format(len(LSTM_test_inputs)))
    print("\nNumber Of Output Test's sequences: {}".format(len(LSTM_test_outputs)))

    # I find it easier to work with numpy arrays rather than pandas dataframes
    # especially as we now only have numerical data
    

    LSTM_test_inputs = [np.array(LSTM_test_input) for LSTM_test_input in LSTM_test_inputs]
    LSTM_test_inputs = np.array(LSTM_test_inputs)
    LSTM_test_inputs = [x.ravel() for x in LSTM_test_inputs]
    LSTM_test_inputs = np.array(LSTM_test_inputs)

    LSTM_test_outputs = [np.array(LSTM_test_output) for LSTM_test_output in LSTM_test_outputs]
    LSTM_test_outputs = np.array(LSTM_test_outputs)
    LSTM_test_outputs = [x.ravel() for x in LSTM_test_outputs]
    LSTM_test_outputs = np.array(LSTM_test_outputs)
    
    

    print ("testing data", LSTM_test_inputs[0])

    
    #####################################
    # Modeling
    #####################################
    
    # initialise model architecture
    bt_model = ModelUtils.build_model(LSTM_test_inputs, output_size=2, neurons_lv1=num_of_neurons_lv1, neurons_lv2=num_of_neurons_lv2,
                                      neurons_lv3=num_of_neurons_lv3,neurons_lv4=num_of_neurons_lv4,activ_func=activ_func, dropout=dropout, loss=loss, optimizer=optimizer)
    
    
   
    print ('weights:')
    # load weights
    bt_model.load_weights(model_path)
    print (bt_model.get_weights())
    print ('weights#')
    
    

    upTrue=0.0
    downTrue=0.0
    upErr=0.0
    downErr=0.0

    Arr=(bt_model.predict_classes(LSTM_test_inputs))
    if Arr[0]==0:
        print ('UP')
        print ('with confidence: ',bt_model.predict(LSTM_test_inputs)[0][0])
    else:
        print ('DOWN')
        print ('with confidence: ',bt_model.predict(LSTM_test_inputs)[0][1])

    
   