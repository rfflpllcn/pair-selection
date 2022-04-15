import csv
from os.path import join
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
from sklearn.cluster import OPTICS
from sklearn.linear_model import LinearRegression
from itertools import combinations
from statsmodels.tsa.stattools import adfuller
from matplotlib import pyplot as plt
import numpy as np
import statistics

from config import ROOT


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


# Function sourced from https://stackoverflow.com/questions/57096574/how-to-apply-the-hurst-exponent-in-python-in-a-rolling-window
def hurst(sp):
    # calculate standard deviation of differenced series using various lags
    lags = range(2, 20)
    tau = [np.sqrt(np.std(np.subtract(sp[lag:], sp[:-lag]))) for lag in lags]

    # calculate Hurst as slope of log-log plot
    m = np.polyfit(np.log(lags), np.log(tau), 1)
    hurst = m[0] * 2.0

    return hurst


# load test returns
data = pd.read_csv(join(ROOT, 'data', 'rets.csv'), index_col=0)
companies = data.columns

# We must transpose the dataframe to ensure it works with sklearn
data = data.transpose()

# Because pca can be affected by the scale of the data we must use StandardScaler
scaler = StandardScaler()
data_scale = scaler.fit_transform(data)

# Run the pca algorithm on the scaled data
pca = decomposition.PCA(n_components=2)
pca.fit(data_scale)
data_pca = pca.transform(data_scale)

# Run the optics algorithm to determine the which stocks are clustered together
clust = OPTICS()
clust.fit(data_pca)

labels = clust.labels_
num_labels = len(labels)

fig, ax2 = plt.subplots()
cmap = get_cmap(num_labels)
colors = [cmap(i) for i in range(num_labels)]

for klass, color in zip(range(0, num_labels), colors):
    Xk = data_pca[clust.labels_ == klass]
    ax2.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
ax2.plot(data_pca[clust.labels_ == -1, 0], data_pca[clust.labels_ == -1, 1], "k+", alpha=0.1)
ax2.set_title("Automatic Clustering\nOPTICS")
plt.show()


# Sort the results of the optics algorithm into a dictionary
# Notice that stocks assigned equal positive numbers are considered clustered together (-1 means no clustering)
results = {}
i = 0
for i in range(0, num_labels):
    if labels[i] not in results.keys():
        results[labels[i]] = [companies[i]]
    else:
        results[labels[i]].append(companies[i])

# Make a list of the pairs we plan to test and which companies data is needed
pairs = []
company = []
for key in results.keys():
    if key != -1:
        comb = combinations(results[key], 2)
        pairs = pairs + list(comb)
        company = company + results[key]

# Download all the necessary data
price_data = {}
for comp in company:
    # Open the companies csv file
    file = open(clean_loc + comp + "_clean.csv")
    value = list(csv.reader(file))

    # Record the log prices of the closing values
    close = []
    for time in value[1:]:
        close.append([float(time[-1])])

    price_data[comp] = close

# Calculate the spread for every pair
pair_spread = []
for pair in pairs:
    # Regression of 0 = logy - nlogx or logy = nlogx
    s1 = price_data[pair[0]]
    s2 = price_data[pair[1]]

    reg1 = LinearRegression().fit(s2, s1)
    reg2 = LinearRegression().fit(s1, s2)

    # Make sure that the regression coefficient is as large as possible
    info = []
    if reg1.coef_[0][0] > reg2.coef_[0][0]:
        info.append(pair[0])
        info.append(pair[1])
        info.append(float(reg1.coef_[0][0]))
    else:
        info.append(pair[1])
        info.append(pair[0])
        info.append(float(reg2.coef_[0][0]))

    # Calculate the spread of the pair
    spread = []
    for i in range(0, len(s1)):
        # logy - nlogx
        spread.append(price_data[info[0]][i][0] - info[2] * price_data[info[1]][i][0])

    info.append(spread)

    pair_spread.append(info)

    # info[0] is y stock. info[1] is x stock. info[2] is regression coefficient. info[3] is spread series

# Check for useful pairs using the hurst exponent
pair_hurst = []
pair_plot = []
for pair in pair_spread:
    # Prepare the spread data
    spread = pair[3][1608:]

    # Calculate the hurst exponent
    H = hurst(spread)

    # Mean reverting/anti-persistent behaviour when H<0.5
    if H < 0.5:
        pair_hurst.append(pair)

# Check for useful pairs using cointegration
pair_coint = []
for pair in pair_hurst:
    # Prepare the spread data
    spread = pair[3][1608:]

    # Check for cointegration at 5%
    p_value = adfuller(spread)[1]

    # If pair satisfies p-value then it moves to the next step
    if p_value <= 0.05 / len(pair_spread):
        pair_coint.append((pair[0], pair[1], spread))

print("Number of pairs to test " + str(len(pair_spread)))
print("Number of pairs after the Hurst Exponent " + str(len(pair_hurst)))
print("Number of pairs after ADF test " + str(len(pair_coint)))
print(pair_coint)

# Plot the spreads of the pairs we found
for pair in pair_coint:
    # Find the mean and std dev of spread
    mean = statistics.mean(pair[2])
    std_dev_above = mean + statistics.stdev(pair[2]) * 1
    std_dev_below = mean - statistics.stdev(pair[2]) * 1

    m_data = [mean] * 150
    sa_data = [std_dev_above] * 150
    sb_data = [std_dev_below] * 150

    plt.plot(pair[2])
    plt.plot(m_data)
    plt.plot(sa_data)
    plt.plot(sb_data)
    plt.title(pair[0] + ' and ' + pair[1])
    plt.show()
