
import matplotlib.pyplot as plt
import datetime
from PIL import Image

class PlotUtils(object):
    '''
    classdocs
    '''

    def __init__(self, params):
        '''
        Constructor
        '''
        
    @staticmethod
    def plotCoinTrend(market_info, logo, coin):   
        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[5, 1]}, figsize=(10, 10))   
        ax1.set_ylabel('Closing Price ($)', fontsize=12)
        ax2.set_ylabel('Volume ($ '+coin+')', fontsize=12)
        ax2.set_yticks([int('%d000000000' % i) for i in range(10)])
        ax2.set_yticklabels(range(10))
        ax1.set_xticks([datetime.date(i, j, 1) for i in range(2013, 2019) for j in [1, 7]])
        ax1.set_xticklabels('')
        ax2.set_xticks([datetime.date(i, j, 1) for i in range(2013, 2019) for j in [1, 7]])
        ax2.set_xticklabels([datetime.date(i, j, 1).strftime('%b %Y')  for i in range(2013, 2019) for j in [1, 7]])
        ax1.plot(market_info['Date'].astype(datetime.datetime), market_info['Open'])
        ax2.bar(market_info['Date'].astype(datetime.datetime).values, market_info['Volume'].values)
        fig.tight_layout()
        fig.figimage(logo, 100, 120, zorder=3, alpha=.5)
        plt.show()
        fig.savefig("Output/"+coin+"Trend.png")
    
    @staticmethod
    def plotCoinTrainingTest(market_info, split_date, coin_im, target, coin):
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))
        ax1.set_xticks([datetime.date(i, j, 1) for i in range(2013, 2019) for j in [1, 7]])
        ax1.set_xticklabels([datetime.date(i, j, 1).strftime('%b %Y')  for i in range(2013, 2019) for j in [1, 7]])
        ax1.plot(market_info[market_info['Date'] < split_date]['Date'].astype(datetime.datetime),
                 market_info[market_info['Date'] < split_date][target],
                 color='#B08FC7')
        ax1.plot(market_info[market_info['Date'] >= split_date]['Date'].astype(datetime.datetime),
                 market_info[market_info['Date'] >= split_date][target], color='#8FBAC8')
        ax1.set_ylabel(coin + ' Price ($)', fontsize=12)
        plt.tight_layout()
        fig.figimage(coin_im.resize((int(coin_im.size[0] * 0.65), int(coin_im.size[1] * 0.65)), Image.ANTIALIAS),
                     350, 40, zorder=3, alpha=.5)
        plt.show()
        fig.savefig("Output/" + coin + "CoinTrainTest.png")
