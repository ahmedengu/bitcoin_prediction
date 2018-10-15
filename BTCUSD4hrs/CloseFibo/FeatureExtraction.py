import pandas as pd
import numpy as np

window_len = 15
timeToPridictInHours=4
timeToTrainInDays=2
num_of_neurons_lv1 = 100
num_of_neurons_lv2 = 100
num_of_neurons_lv3 = 25
num_of_neurons_lv4 = 10
model_path = "bt_model.h5"
model_weights_path = "bt_model_weights.h5"
dataset_path='BITFINEX_SPOT_BTC_USD_'+str(timeToPridictInHours)+'HRS.csv'
result_Path=str(timeToPridictInHours)+"HR Result.csv"


def ICHIMOKU(df):
    # Turning Line
    period9_high = pd.Series(df['bt_high'][::-1].rolling(9).max())
    period9_low = pd.Series(df['bt_low'][::-1].rolling(9).min())
    TL = pd.Series((period9_high + period9_low) / 2)[::-1]
    
    # Standard Line
    period26_high = df['bt_high'][::-1].rolling(window=26, center=False).max()
    period26_low = df['bt_low'][::-1].rolling(window=26, center=False).min()
    SL = pd.Series((period26_high + period26_low) / 2)[::-1]
    
    # Leading Span 1
    ICHIMOKU_SPAN1 = pd.Series(((TL + SL) / 2).shift(26), name='bt_ICHIMOKU_SPAN1')[::-1]
    df = df.join(ICHIMOKU_SPAN1.fillna(0))
    
    # Leading Span 2
    period52_high = df['bt_high'].rolling(window=52, center=False).max()
    period52_low = df['bt_low'].rolling(window=52, center=False).min()
    ICHIMOKU_SPAN2 = pd.Series(((period52_high + period52_low) / 2).shift(26), name='bt_ICHIMOKU_SPAN2')[::-1]
    df = df.join(ICHIMOKU_SPAN2.fillna(0))
    
    # The most current closing price plotted 22 time periods behind (optional)
    CHIKOU_SPAN = pd.Series(df['bt_close'].shift(-22), name='bt_CHIKOU_SPAN')[::-1]
    df = df.join(CHIKOU_SPAN.fillna(0))
    return df


def PSAR(df, iaf = 0.02, maxaf = 0.2):
    length = len(df)
    bt_PSAR = pd.Series(df['bt_close'][0:len(df['bt_close'])],name='bt_PSAR')[::-1]
    psarbull = [None] * length
    psarbear = [None] * length
    bull = True
    af = iaf
    ep = df['bt_low'][0]
    hp = df['bt_high'][0]
    lp = df['bt_low'][0]
    for i in range(2,length):
        if bull:
            bt_PSAR[i] = bt_PSAR[i - 1] + af * (hp - bt_PSAR[i - 1])
        else:
            bt_PSAR[i] = bt_PSAR[i - 1] + af * (lp - bt_PSAR[i - 1])
        reverse = False
        if bull:
            if df['bt_low'][i] < bt_PSAR[i]:
                bull = False
                reverse = True
                bt_PSAR[i] = hp
                lp = df['bt_low'][i]
                af = iaf
        else:
            if df['bt_high'][i] > bt_PSAR[i]:
                bull = True
                reverse = True
                bt_PSAR[i] = lp
                hp = df['bt_high'][i]
                af = iaf
        if not reverse:
            if bull:
                if df['bt_high'][i] > hp:
                    hp = df['bt_high'][i]
                    af = min(af + iaf, maxaf)
                if df['bt_low'][i - 1] < bt_PSAR[i]:
                    bt_PSAR[i] = df['bt_low'][i - 1]
                if df['bt_low'][i - 2] < bt_PSAR[i]:
                    bt_PSAR[i] = df['bt_low'][i - 2]
            else:
                if df['bt_low'][i] < lp:
                    lp = df['bt_low'][i]
                    af = min(af + iaf, maxaf)
                if df['bt_high'][i - 1] > bt_PSAR[i]:
                    bt_PSAR[i] = df['bt_high'][i - 1]
                if df['bt_high'][i - 2] > bt_PSAR[i]:
                    bt_PSAR[i] = df['bt_high'][i - 2]
        if bull:
            psarbull[i] = bt_PSAR[i]
        else:
            psarbear[i] = bt_PSAR[i]
    df = df.join(bt_PSAR.fillna(0))
    return df


#Moving Average
def MA(df, n):
    MA = pd.Series(df['bt_close'][::-1].rolling(n).mean(), name = 'bt_MA_' + str(n))[::-1]#[::-1] mean rows n col ka last elm
    df = df.join(MA.fillna(0))
    return df

#Exponential Moving Average
def EMA(df, n):
    EMA = pd.Series(df['bt_close'][::-1].ewm(span = n, min_periods = n - 1).mean(), name = 'bt_EMA_' + str(n))[::-1]
    df = df.join(EMA.fillna(0))
    return df


#Momentum
def MOM(df, n):
    M = pd.Series(df['bt_close'][::-1].diff(n), name = 'bt_MOM_' + str(n))[::-1]
    df = df.join(M.fillna(0))
    return df

#Rate of Change
def ROC(df, n):
    M = df['bt_close'][::-1].diff(n - 1)
    N = df['bt_close'][::-1].shift(n - 1)
    ROC = pd.Series(M / N, name = 'bt_ROC_' + str(n))[::-1]
    df = df.join(ROC.fillna(0))
    return df

#Average True Range  
def ATR(df, n):  
    i = 0  
    TR_l = [0]  
    while i < df.index[-1]:
        TR = max(df.at[i + 1, 'bt_high'], df.at[i, 'bt_close'])\
        - min(df.at[i + 1, 'bt_low'], df.at[i, 'bt_close'])
        TR_l.append(TR)  
        i = i + 1  
    TR_s = pd.Series(TR_l)  
    ATR = pd.Series(TR_s[::-1].ewm( span = n, min_periods = n).mean(),\
                    name = 'bt_ATR_' + str(n))[::-1]  
    df = df.join(ATR.fillna(0))  
    return df

#Bollinger Bands  
def BBANDS(df, n):  
    MA = pd.Series(df['bt_close'][::-1].rolling(n).mean()) [::-1]
    MSD = pd.Series(df['bt_close'][::-1].rolling(n).std())  [::-1]
    b1 = 4 * MSD / MA  
    B1 = pd.Series(b1[::-1], name = 'bt_BollingerB1_' + str(n))  [::-1]
    df = df.join(B1.fillna(0))  
    b2 = (df['bt_close'][::-1] - MA + 2 * MSD) / (4 * MSD)  [::-1]
    B2 = pd.Series(b2[::-1], name = 'bt_BollingerB2_' + str(n))  [::-1]
    df = df.join(B2.fillna(0))  
    return df

#Pivot Points, Supports and Resistances  
def PPSR(df):  
    PP = pd.Series((df['bt_high'] + df['bt_low'] + df['bt_close']) / 3)
    R1 = pd.Series(2 * PP - df['bt_low'])
    S1 = pd.Series(2 * PP - df['bt_high'])
    R2 = pd.Series(PP + df['bt_high'] - df['bt_low'])
    S2 = pd.Series(PP - df['bt_high'] + df['bt_low'])
    R3 = pd.Series(df['bt_high'] + 2 * (PP - df['bt_low']))
    S3 = pd.Series(df['bt_low'] - 2 * (df['bt_high'] - PP))
    psr = {'bt_PP':PP, 'bt_R1':R1, 'bt_S1':S1, 'bt_R2':R2,\
           'bt_S2':S2, 'bt_R3':R3, 'bt_S3':S3}  
    PSR = pd.DataFrame(psr)  
    df = df.join(PSR.fillna(0))  
    return df

#Stochastic oscillator %K  
def STOK(df):  
    SOk = pd.Series((df['bt_close'] - df['bt_low']) / \
                    (df['bt_high'] - df['bt_low']), name = 'bt_SOK')
    df = df.join(SOk.fillna(0))  
    return df

# Stochastic Oscillator, EMA smoothing, nS = slowing (1 if no slowing)  
def STO(df,  nK, nD, nS=1):  
    SOk = pd.Series((df['bt_close'][::-1] - df['bt_low'][::-1].rolling(nK).min()) / (df['bt_high'][::-1].rolling(nK).max() - df['bt_low'][::-1].rolling(nK).min()), name = 'bt_SOK_'+str(nK)) [::-1]
    SOd = pd.Series(SOk[::-1].ewm(ignore_na=False, span=nD, min_periods=nD-1, adjust=True).mean(), name = 'bt_SOD_'+str(nD))  [::-1]
    SOk = SOk.ewm(ignore_na=False, span=nS, min_periods=nS-1, adjust=True).mean()  
    SOd = SOd.ewm(ignore_na=False, span=nS, min_periods=nS-1, adjust=True).mean()   
    df = df.join(SOk.fillna(0))
    df = df.join(SOd.fillna(0))  
    return df  

#Trix  
def TRIX(df, n):  
    EX1 = df['bt_close'][::-1].ewm(span = n, min_periods = n - 1).mean()[::-1]
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
        UpMove = df.at[i + 1, 'bt_high'] - df.at[i, 'bt_high']
        DoMove = df.at[i, 'bt_low'] - df.at[i + 1, 'bt_low']
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
        TR = max(df.at[i + 1, 'bt_high'], df.at[i, 'bt_close']) - min(df.at[i + 1, 'bt_low'], df.at[i, 'bt_close'])
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
    EMAfast = pd.Series(df['bt_close'][::-1].ewm( span = n_fast, min_periods = n_slow - 1).mean()) [::-1]
    EMAslow = pd.Series(df['bt_close'][::-1].ewm( span = n_slow, min_periods = n_slow - 1).mean()) [::-1]
    MACD = pd.Series(EMAfast - EMAslow, name = 'bt_MACD_' + str(n_fast) + '_' + str(n_slow))  
    MACDsign = pd.Series(MACD[::-1].ewm( span = 9, min_periods = 8).mean(), name = 'bt_MACDsign_' + str(n_fast) + '_' + str(n_slow))  [::-1]
    MACDdiff = pd.Series(MACD - MACDsign, name = 'bt_MACDdiff_' + str(n_fast) + '_' + str(n_slow))  
    df = df.join(MACD.fillna(0))  
    df = df.join(MACDsign.fillna(0))  
    df = df.join(MACDdiff.fillna(0))  
    return df

#Mass Index  
def MassI(df):  
    Range = df['bt_high'] - df['bt_low']
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
        Range = max(df.at[i + 1, 'bt_high'], df.at[i, 'bt_close']) - min(df.at[i + 1, 'bt_low'], df.at[i, 'bt_close'])
        TR.append(Range)  
        i = i + 1  
    i = 0  
    VM = [0]  
    while i < df.index[-1]:  
        Range = abs(df.at[i + 1, 'bt_high'] - df.at[i, 'bt_low']) - abs(df.at[i + 1, 'bt_low'] - df.at[i, 'bt_high'])
        VM.append(Range)  
        i = i + 1  
    VI = pd.Series(pd.Series(VM)[::-1].rolling( n).sum() / pd.Series(TR)[::-1].rolling( n).sum(), name = 'bt_Vortex_' + str(n))  [::-1]
    df = df.join(VI.fillna(0))  
    return df

#KST Oscillator  
def KST(df): 
    r1 = 10
    r2 = 15
    r3 = 20
    r4 = 30
    n1 = 10
    n2 = 10
    n3 = 10
    n4 = 15
    M = df['bt_close'][::-1].diff(r1 - 1)[::-1]
    N = df['bt_close'][::-1].shift(r1 - 1)  [::-1]
    ROC1 = M / N  
    M = df['bt_close'][::-1].diff(r2 - 1) [::-1]
    N = df['bt_close'][::-1].shift(r2 - 1)[::-1]
    ROC2 = M / N  
    M = df['bt_close'][::-1].diff(r3 - 1) [::-1]
    N = df['bt_close'][::-1].shift(r3 - 1)[::-1]
    ROC3 = M / N  
    M = df['bt_close'][::-1].diff(r4 - 1) [::-1]
    N = df['bt_close'][::-1].shift(r4 - 1)[::-1]
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
        UpMove = df.at[i + 1, 'bt_high'] - df.at[i, 'bt_high']
        DoMove = df.at[i, 'bt_low'] - df.at[i + 1, 'bt_low']
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
    M = pd.Series(df['bt_close'].diff(1))
    aM = abs(M)  
    EMA1 = pd.Series(M[::-1].ewm( span = r, min_periods = r - 1).mean())  [::-1]
    aEMA1 = pd.Series(aM[::-1].ewm( span = r, min_periods = r - 1).mean())  [::-1]
    EMA2 = pd.Series(EMA1[::-1].ewm( span = s, min_periods = s - 1).mean()) [::-1] 
    aEMA2 = pd.Series(aEMA1[::-1].ewm( span = s, min_periods = s - 1).mean()) [::-1]  
    TSI = pd.Series(EMA2 / aEMA2, name = 'bt_TSI_' + str(r) + '_' + str(s))  
    df = df.join(TSI.fillna(0))
    return df

# fibonnaci_Algo
def fibonnaci_algo(df):
    # select all the dataset
    df_subclose = df.bt_close
    close_p = df_subclose.tolist()  # transform df to list
    len_p = len(close_p)  # length of dataset
    ema5 = pd.Series.ewm(df_subclose, span=5).mean().tolist()
    ema20 = pd.Series.ewm(df_subclose, span=20).mean().tolist()
    ret_level1 = [0, 0]
    ret_level2 = [0, 0]
    ret_level3 = [0, 0]
    ext_level1 = [0, 0]
    ext_level2 = [0, 0]
    ext_level3 = [0, 0]
    for i in range(2, len_p):
        price_min = min(close_p[:i])
        price_max = max(close_p[:i])
        diff = price_max - price_min
        if ema5[i] > ema20[i]:
            # fibonnaci retracement and extensions for downward move
            ret_level1.append(price_min + 0.236 * diff)
            ret_level2.append(price_min + 0.382 * diff)
            ret_level3.append(price_min + 0.618 * diff)
            ext_level1.append(price_min - 0.236 * diff)
            ext_level2.append(price_min - 0.382 * diff)
            ext_level3.append(price_min - 0.618 * diff)
        else:
            # find minimum and max closing price before this point
            # fibonnaci retracement and extensions for upward move
            ret_level1.append(price_max - 0.236 * diff)
            ret_level2.append(price_max - 0.382 * diff)
            ret_level3.append(price_max - 0.618 * diff)
            ext_level1.append(price_max + 0.236 * diff)
            ext_level2.append(price_max + 0.382 * diff)
            ext_level3.append(price_max + 0.618 * diff)
    
    df = df.join(pd.Series(ret_level1, name='bt_ret_level1').fillna(0))
    df = df.join(pd.Series(ret_level2, name='bt_ret_level2').fillna(0))
    df = df.join(pd.Series(ret_level3, name='bt_ret_level3').fillna(0))
    df = df.join(pd.Series(ext_level1, name='bt_ext_level1').fillna(0))
    df = df.join(pd.Series(ext_level2, name='bt_ext_level2').fillna(0))
    df = df.join(pd.Series(ext_level3, name='bt_ext_level3').fillna(0))
    return df




# It reads the file and extract all the features
def extractFeaturesFromData(isTraining=False):
    bt_market_info = pd.read_csv(dataset_path)
    if isTraining:
        bt_market_info = bt_market_info.tail(window_len)
        bt_market_info.index = pd.Index(data=[i for i in range(len(bt_market_info))])
    for h in range(len(bt_market_info)):
        if (np.isnan(bt_market_info.market_cap[h])):
            # print('yes',h)
            mk1 = bt_market_info.market_cap[h - 1]
            bt_market_info.market_cap[h] = mk1
    print('Preprocessing start')
    
    # convert the date string to the correct date format
    bt_market_info = bt_market_info.assign(Date=pd.to_datetime(bt_market_info['date']))
    # Put bt_ in front of column names
    bt_market_info.columns = [bt_market_info.columns[0]] + ['bt_' + i for i in bt_market_info.columns[1:]]
    
    ###Feature Extraction
    for coins in ['bt_']:
        kwargs = {coins + 'day_diff': lambda x: (x[coins + 'close'] - x[coins + 'open']) / x[coins + 'open']}
        bt_market_info = bt_market_info.assign(**kwargs)
    
    for coins in ['bt_']:
        kwargs = {coins + 'volatility': lambda x: (x[coins + 'high'] - x[coins + 'low']) / (x[coins + 'open'])}
        bt_market_info = bt_market_info.assign(**kwargs)
    
    bt_market_info = MA(bt_market_info, window_len)
    bt_market_info = EMA(bt_market_info, window_len)
    bt_market_info = MOM(bt_market_info, window_len)
    bt_market_info = ROC(bt_market_info, window_len)
    bt_market_info = ATR(bt_market_info, window_len)
    bt_market_info = BBANDS(bt_market_info, window_len)
    bt_market_info = PPSR(bt_market_info)
    bt_market_info = STOK(bt_market_info)
    bt_market_info = STO(bt_market_info, window_len, window_len)
    bt_market_info = TRIX(bt_market_info, window_len)
    bt_market_info = ADX(bt_market_info, window_len, window_len * 2)
    bt_market_info = MACD(bt_market_info, window_len * 2, window_len)
    bt_market_info = MassI(bt_market_info)
    bt_market_info = Vortex(bt_market_info, window_len)
    bt_market_info = KST(bt_market_info)
    bt_market_info = RSI(bt_market_info, window_len)
    bt_market_info = TSI(bt_market_info, window_len * 2, window_len)
    bt_market_info = ICHIMOKU(bt_market_info)
    bt_market_info = PSAR(bt_market_info)
    bt_market_info = fibonnaci_algo(bt_market_info)
    
    wl = str(window_len)
    model_data = bt_market_info[['date'] + [coin + metric for coin in ['bt_'] \
                                            for metric in
                                            ['close', 'volatility', 'day_diff', 'high', 'low', 'open', 'volume', \
                                             'market_cap', 'MA_' + wl, 'EMA_' + wl, 'MOM_' + wl, 'ROC_' + wl,
                                             'ATR_' + wl, 'BollingerB1_' + wl, 'BollingerB2_' + wl, \
                                             'PP', 'R1', 'S1', 'R2', 'S2', 'R3', 'S3', 'SOK', 'SOK_' + wl, 'SOD_' + wl,
                                             'Trix_' + wl, 'ADX_' + wl, \
                                             'MACD_' + str(window_len * 2) + '_' + wl,
                                             'MACDsign_' + str(window_len * 2) + '_' + wl,
                                             'MACDdiff_' + str(window_len * 2) + '_' + wl, 'Mass_Index', 'Vortex_' + wl,
                                             'KST', 'RSI_' + wl, 'TSI_' + str(window_len * 2) + '_' + wl,
                                             'ICHIMOKU_SPAN1', 'ICHIMOKU_SPAN2', 'CHIKOU_SPAN', 'PSAR'
                                             ,'ret_level1','ret_level2','ret_level3',
                                             'ext_level1','ext_level2','ext_level3'
                                             ]]]
    return model_data
