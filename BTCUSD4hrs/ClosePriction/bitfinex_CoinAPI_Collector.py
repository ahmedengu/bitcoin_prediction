import datetime
import json
import threading
import pandas as pd
import requests
import websocket  # version 0.48.0
from FeatureExtraction import dataset_path,timeToPridictInHours
import time


class BitfinexTradesDataCollector(threading.Thread):
    # supported periods format can be found here: http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
    # callback return a dataframe, columns: ['price_open', 'price_high', 'price_low', 'price_close', 'volume_traded', 'trades_count', 'time_close', 'time_open', 'time_period_start', 'time_period_end', 'market_cap']

    def __init__(self, threadID,base, quote, period, callback):
        self.threadID=threadID
        self.symbol = base + quote
        self.quote = quote.upper()
        self.base = base.upper()
        self.period = period
        self.callback = callback
        self.ws = None
        self.transactions = pd.DataFrame(columns=['date', 'price', 'quantity'])
        self.today = datetime.datetime.utcnow().date()
        self.supply = None
        self.update_supply()
        self.filename = "dataset.csv"
    
        threading.Thread.__init__(self)

    def schedule_callback(self):
        delay = (pd.date_range(datetime.datetime.utcnow(), periods=1, freq=self.period).ceil(self.period)[
                     0] - datetime.datetime.utcnow()).total_seconds()
    
        self.callback_thread = threading.Timer(delay, self.period_callback)
        self.callback_thread.start()

    def period_callback(self):
        try:
            if (len(self.transactions) > 0):
                df = self.transactions
                self.transactions = pd.DataFrame(columns=['date', 'price', 'quantity'])
            
                df.index = pd.DatetimeIndex(data=pd.to_datetime(df['date'], unit='ms'))
            
                df_date = pd.DataFrame(df['date'].resample(self.period).ohlc())
                p = pd.DataFrame(df['price'].resample(self.period).ohlc())
                p.columns = ['price_open', 'price_high', 'price_low', 'price_close']
                p['volume_traded'] = pd.DataFrame(df['quantity'].abs().resample(self.period).sum())
                time_open = pd.to_datetime(df_date['open'], unit='ms')
                p['time_period_start'] = time_open.dt.floor(self.period)
                p['market_cap'] = p['price_close'] * self.supply
                data = p['time_period_start'].dt.strftime('%Y-%m-%dT%H:%M:%S')
            
                p.index = data
                self.callback(p)
                self.writeInCsvFile(p)
        except Exception as e:
            print(e)
    
        self.schedule_callback()
        if self.today != datetime.datetime.utcnow().date():
            self.today = datetime.datetime.utcnow().date()
            self.update_supply()

    def update_supply(self):
        CC_r = requests.request('GET',
                                'https://min-api.cryptocompare.com/data/pricemultifull?fsyms=' + self.base + '&tsyms=' + self.quote)
        if 'Error' not in CC_r.text:
            self.supply = json.loads(CC_r.text)['RAW'][self.base][self.quote]['SUPPLY']

    def run(self):
        print("{1} :: Starting a socket connection for {0} Trades".format(self.symbol, datetime.datetime.utcnow().strftime(
            "%Y-%b-%d %H:%M:%S")))
        self.schedule_callback()
        self.initiate_socket_connection()

    def initiate_socket_connection(self):
        self.ws = websocket.WebSocketApp('wss://api.bitfinex.com/ws/2',
                                         on_message=self.message_processor,
                                         on_error=self.error_processor,
                                         on_close=self.socket_closer)
    
        self.ws.on_open = self.socket_opener
        self.ws.run_forever()

    def socket_opener(self, ws):
        print("{1} :: Opening the socket connection for {0}".format(self.symbol, datetime.datetime.utcnow().strftime(
            "%Y-%b-%d %H:%M:%S")))
        ws.send('{"event": "subscribe", "channel": "trades", "symbol": "' + self.symbol + '"}')
        return

    def message_processor(self, ws, message):
        try:
            message = json.loads(message)
            if message[1] == "tu":  # tu: trade execution update, te:trade executed
                self.transactions = self.transactions.append(pd.Series({
                    'date': message[2][1],
                    'price': message[2][3],
                    'quantity': message[2][2]
                }), ignore_index=True)
        except Exception as e:
            if e.args[0] == 1:
                print(message)
            else:
                print(e)

    def error_processor(self, ws, message):
        print("{1} :: Error on the socket connection for {0}".format(self.symbol, datetime.datetime.utcnow().strftime(
            "%Y-%b-%d %H:%M:%S")))
        print(message)
        return

    def socket_closer(self, ws):
        print("{1} :: Closing the socket connection for {0}".format(self.symbol, datetime.datetime.utcnow().strftime(
            "%Y-%b-%d %H:%M:%S")))
        return

    def stop(self):
        self.ws.close()
        self.callback_thread.cancel()
        self._stop()

    def writeInCsvFile(self, df):
        del df['time_period_start']
        with open(dataset_path, 'a') as f:
            df.to_csv(f, header=False)

if __name__ == "__main__":
    # x is a dataframe, columns: ['price_open', 'price_high', 'price_low', 'price_close', 'volume_traded', 'trades_count', 'time_close', 'time_open', 'time_period_start', 'time_period_end', 'market_cap']
    callback = (lambda x: print(x.iloc[0]))
    market="BTC"
    trade_connections = {}
    threadID=1
    timediference=str(timeToPridictInHours)+'H'
    while True:
        if (market in trade_connections) and not trade_connections[market].is_alive():
            print("{1} :: Restarting the thread for market {0}".format(market, datetime.datetime.utcnow().strftime("%Y-%b-%d %H:%M:%S")))
            del trade_connections[market]
            trade_connections[market] = BitfinexTradesDataCollector(threadID,market, 'USD', timediference, callback)
            threadID += 1
            trade_connections[market].start()
            time.sleep(10)

        elif (market not in trade_connections):# and trade_connections[market].is_alive():
            trade_connections[market] = BitfinexTradesDataCollector(threadID,market, 'USD', timediference, callback)
            threadID += 1
            trade_connections[market].start()
            time.sleep(10)
        time.sleep(30)


