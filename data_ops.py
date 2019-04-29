import fix_yahoo_finance as yf
import matplotlib.pyplot as plt

def get_datasets(stock_name):
    if (stock_name == "Amazon"):
        ticker = yf.Ticker("AMZN")
        dataset = ticker.history(period='max')
        dataset.isna().any()
        return [dataset]
    elif (stock_name == "Google"):
        ticker = yf.Ticker("GOOG")
        dataset = ticker.history(period='max')
        dataset.isna().any()
        return [dataset]
    elif (stock_name == "Apple"):
        ticker = yf.Ticker("AAPL")
        dataset = ticker.history(period='max')
        dataset.isna().any()
        return [dataset]
    elif (stock_name == "Microsoft"):
        ticker = yf.Ticker("MSFT")
        dataset = ticker.history(period='max')
        dataset.isna().any()
        return [dataset]
    elif (stock_name == "AAL"):
        ticker = yf.Ticker("AAL")
        dataset = ticker.history(period='max')
        dataset.isna().any()
        return [dataset]
    elif (stock_name == "Tesla"):
        ticker = yf.Ticker("TSLA")
        dataset = ticker.history(period='max')
        dataset.isna().any()
        return [dataset]
    elif (stock_name == "Bitcoin"):
        ticker = yf.Ticker("BTC-USD")
        dataset = ticker.history(period='max')
        dataset.isna().any()
        return [dataset]

    elif (stock_name== "Aut"):
        symbols = ["VOE.VI"]
        datasets = []
        for s in symbols:
            ticker = yf.Ticker(s)
            dataset = ticker.history(period='max')
            dataset.isna().any()
            datasets.append(dataset)
        return datasets

    elif (stock_name== "MixedTech"):
        symbols = ["AMZN","TSLA","AAPL","GOOG","MSFT"]
        datasets = []
        for s in symbols:
            ticker = yf.Ticker(s)
            dataset = ticker.history(period='max')
            dataset.isna().any()
            datasets.append(dataset)
        return datasets

    elif (stock_name== "Mixed"):
        symbols = ["GE","VWAGY","BMWYY","F"]
        #symbols = ["TSLA","AMD"]
        datasets = []
        for s in symbols:
            ticker = yf.Ticker(s)
            dataset = ticker.history(period='max')
            dataset.isna().any()
            datasets.append(dataset)
        return datasets

    elif (stock_name== "MixedCrypto"):
        symbols = ["BTC-USD","ETH-USD"]
        datasets = []
        for s in symbols:
            ticker = yf.Ticker(s)
            dataset = ticker.history(period='max')
            dataset.isna().any()
            datasets.append(dataset)
        return datasets
    elif (stock_name== "SinTest"):
        datasets = []

        return datasets
    elif (stock_name== "eko"):
        symbols = ["AIXA.DE"]
        datasets = []
        for s in symbols:
            ticker = yf.Ticker(s)
            dataset = ticker.history(period='max')
            dataset.isna().any()
            datasets.append(dataset)
        return datasets
    else:
        ticker = yf.Ticker(stock_name)
        dataset = ticker.history(period='max')
        dataset.isna().any()
        return [dataset]

'''***************************************PLOTTING*****************************************************'''

def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    #plt.legend()
    plt.show()


def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
	# Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        #plt.legend()
    plt.show()