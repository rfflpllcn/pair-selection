"""
data = yf.download(  # or pdr.get_data_yahoo(...
        # tickers list or string as well
        tickers = "SPY AAPL MSFT",

        # use "period" instead of start/end
        # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        # (optional, default is '1mo')
        period = "ytd",

        # fetch data by interval (including intraday if period < 60 days)
        # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        # (optional, default is '1d')
        interval = "1m",

        # group by ticker (to access via data['SPY'])
        # (optional, default is 'column')
        group_by = 'ticker',

        # adjust all OHLC automatically
        # (optional, default is False)
        auto_adjust = True,

        # download pre/post regular market hours data
        # (optional, default is False)
        prepost = True,

        # use threads for mass downloading? (True/False/Integer)
        # (optional, default is True)
        threads = True,

        # proxy URL scheme use use when downloading?
        # (optional, default is None)
        proxy = None
    )

"""
# Import the necessary modules
from os.path import join, exists
import pandas as pd
import yfinance as yf
import pickle

from config import ROOT


class PriceDownloader:
    """
    Usage:
        downloader = PriceDownloader("sp500", start_date='2021-01-01', end_date='2022-01-01', interval='1d')
        prices = downloader.load()

    """

    def __init__(self, market, start_date=None, end_date=None, interval=None):
        """

        :param market:
        :param start_date:
        :param end_date:
        :param interval (str): Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        """
        # self.date_range = pd.date_range(start_date, end_date, periods, freq)
        self._assets = None
        self.fpath = None

        self.market = market
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval

    @property
    def market(self):
        return self.market

    @market.setter
    def market(self, value):
        if value == "sp500":
            sp_assets = pd.read_html(
                'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
            self.assets = sp_assets.Symbol.tolist()
            fname = 'sp_components_data'
        else:
            self.assets = [value] if isinstance(value, str) else value
            fname = 'sp_index_data'
            # raise NotImplementedError("market {} not supported".format(value))

        self.fpath = join(ROOT, 'data', '{}.pkl'.format(fname))

    @property
    def assets(self):
        return self._assets

    @assets.setter
    def assets(self, value):
        self._assets = value

    def download(self):
        # Download historical data to a multi-index DataFrame
        if self.assets is None:
            raise ValueError("assets = None should not happen")

        try:
            data = yf.download(self.assets, start=self.start_date, end=self.end_date, as_panel=False, group_by="ticker",
                               interval=self.interval)
            data.to_pickle(self.fpath)
            print('Data saved at {}'.format(self.fpath))
        except ValueError:
            print('Failed download, try again.')
            data = None

    def load(self, level='Adj Close', load_if_exists=True, exclude_tickers=None, exclude_tickers_with_nans=False):
        if (not exists(self.fpath)) or (not load_if_exists):
            self.download()

        file = open(self.fpath, 'rb')
        prices = pickle.load(file)

        # if isinstance(prices.columns, pd.MultiIndex):
        if len(self.assets) > 1:
            out = prices.iloc[:, prices.columns.get_level_values(1) == level]
            out.columns = out.columns.get_level_values(0)
        else:
            out = prices[level].to_frame()
            out.columns = self.assets

        if exclude_tickers is not None:
            exclude_tickers = [exclude_tickers] if isinstance(exclude_tickers, str) else exclude_tickers
            out = out[[_c for _c in out.columns if _c not in exclude_tickers]]

        if exclude_tickers_with_nans:
            out = out[out.isnull().any(axis=1).T == False]
        return out


if __name__ == '__main__':
    from yfinance.shared import _ERRORS

    downloader = PriceDownloader("sp500", start_date='2021-01-01', end_date='2022-01-01', interval='1d')
    prices = downloader.load(load_if_exists=True, exclude_tickers=['BRK.B', 'CEG', 'BF.B'])

    print("ERROR", _ERRORS)

    print()
