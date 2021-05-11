import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Process arguments
parser = argparse.ArgumentParser()
parser.add_argument('source')
args = parser.parse_args()

# Convert raw data into a list of measurements
#   series : benchmark
#   parser : parser
#   bytes  : input size
#   iters  : sample iterations
#   value  : duration
data = []

with open(args.source, 'r') as f:
    for line in f.readlines():
        msg = json.loads(line)
        if msg['reason'] != 'benchmark-complete':
            continue

        name = msg['id'].split(sep='/')
        series = name[0]
        parser = name[1]
        bytes = msg['throughput'][0]['per_iteration']
        iters = msg['iteration_count']
        values = msg['measured_values']

        for iter_count, value in zip(iters, values):
            data.append({'series': series, 'parser': parser, 'bytes': bytes, 'iters': iter_count, 'value': value})

# Aggregate data into a frame and set the index
df = pd.DataFrame(data)
df.set_index(['series', 'parser', 'bytes'], inplace=True)

# DEBUG: some simple stats
print(df.groupby(level=['series','parser','bytes']).aggregate(['min', 'mean', 'max']))


# Estimate gradient and 5σ for linear regression.
def regress(df):
    σ2x = df['iters'].var()
    if σ2x == 0:
        print('Warning: sample used only one iteration count')
        return pd.Series({'β': df['value'].mean(), 'σβ_5': 5 * df['value'].std()})
    else:
        X = np.array(df['iters']).reshape(-1, 1)
        y = df['value']
        model = LinearRegression()
        model.fit(X, y)
        β = model.coef_[0]
        σβ_5 = 5 * np.sqrt(((y - model.predict(X))**2).sum() / (len(y) - 2) / σ2x)
        return pd.Series({'β': β, 'σβ_5': σβ_5})

# Calculate the linear regression
regress = df.groupby(level=['series', 'parser', 'bytes']).apply(regress)
print(regress)

# Plot each benchmark separately
for series, data in regress.groupby(level='series'):
    _ , ax = plt.subplots(figsize=(6, 4), dpi=288)

    for parser, data in data.groupby(level='parser'):
        data = data.reset_index()
        x = np.logspace(np.log(data['bytes'].min()), np.log(data['bytes'].max()), base=np.e)

        model = LinearRegression()
        model.fit(np.log(np.array(data['bytes']).reshape(-1, 1)), np.log(data['β']))

        p = ax.errorbar(data['bytes'], data['β'] / 1000, data['σβ_5'] / 1000, marker='o', linestyle='none', label=parser)
        ax.plot(x, np.exp(model.predict(np.log(x).reshape(-1, 1))) / 1000, color=p[0].get_color())

    ax.set_xlabel('Input Size (B)')
    ax.set_ylabel('Parse Time (μs)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()

    plt.savefig(series+'.png')
