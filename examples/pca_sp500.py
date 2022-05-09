"""
replication of the S&P 500 via PCA

the PCA portfolio doesn’t replicate the S&P500 exactly, since the S&P500 is a market-capitalisation weighted average
of the 500 stocks, while the weights in the PCA portfolio is influenced by the explained variance.

the above difference explains why by increasing the number of pca components, the replication gets worse.

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler


def cov2corr(cov):
    vol = np.sqrt(np.diag(cov))
    corr = cov / np.outer(vol, vol)

    return corr


from config import ROOT
from yahoo import PriceDownloader


if __name__ == "__main__":
    start_date='2021-01-01'
    end_date = '2022-01-01'
    interval = '1d'

    # load prices (index)
    downloader = PriceDownloader("SPY", start_date=start_date, end_date=end_date, interval=interval)
    price_index = downloader.load(load_if_exists=True, exclude_tickers=['BRK.B', 'CEG', 'BF.B'])
    rets_index = price_index.pct_change()

    # load prices (components)
    downloader = PriceDownloader("sp500", start_date=start_date, end_date=end_date, interval=interval)
    price_data = downloader.load(load_if_exists=True, exclude_tickers=['BRK.B', 'CEG', 'BF.B'])
    rets = price_data.pct_change()

    rets.dropna(axis=0, how='all', inplace=True)  # drop first row (NA)
    rets.dropna(axis=1, how='any', inplace=True)

    companies = price_data.columns

    scaler = StandardScaler()
    xy_ = scaler.fit_transform(rets)

    corr = cov2corr(np.cov(xy_.T))
    evals, evects = np.linalg.eig(corr)

    # pca on xy
    pca = decomposition.PCA(5)
    xy_pca = pca.fit_transform(xy_)  # = np.dot(xy_, pca.components_)

    fig, ax = plt.subplots(4, 1)
    for i in range(4):
        weights = (np.abs(pca.components_[:i+1,:].sum(axis=0)))/np.sum(np.abs(pca.components_[:i+1,:].sum(axis=0)))

        rs_df = pd.concat([(weights*rets).sum(1), rets_index], 1)
        rs_df.columns = ["PCA Portfolio", "S&P500"]

        sp_reconstructed = rs_df.cumsum().apply(np.exp)

        sp_reconstructed.plot(ax=ax[i])
        ax[i].get_xaxis().set_visible(False)

    plt.show()

