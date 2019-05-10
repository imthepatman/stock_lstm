import fix_yahoo_finance as yf
import matplotlib.pyplot as plt

def get_datasets(stock_name,data_columns):

    if (stock_name== "MixedTech"):
        symbols = ["Amazon","Tesla","Apple","Google","Microsoft"]
        datasets = []
        for s in symbols:
            datasets.append(get_single_dataset(s,data_columns))
        return datasets

    elif (stock_name== "crypto"):
        symbols = ["Bitcoin","Ethereum"]
        datasets = []
        for s in symbols:
            datasets.append(get_single_dataset(s,data_columns))
        return datasets

    elif (stock_name== "eko"):
        symbols = ["Aixtron","GAIA","SunOpta"]
        datasets = []
        for s in symbols:
            datasets.append(get_single_dataset(s,data_columns))
        return datasets

    elif (stock_name== "portfolio"):
        symbols = ["Infineon","Aixtron","GAIA","SunOpta"]
        datasets = []
        for s in symbols:
            datasets.append(get_single_dataset(s,data_columns))
        return datasets
    elif(stock_name=="SemiCon"):
        symbols = ["Aixtron", "Infineon"]
        datasets = []
        for s in symbols:
            datasets.append(get_single_dataset(s,data_columns))
        return datasets

    elif (stock_name== "SinTest"):
        datasets = []

        return datasets

    else:
        return [get_single_dataset(stock_name,data_columns)]



def get_single_dataset(stock_name,data_columns):
    if (stock_name == "Amazon"):
        ticker_name = "AMZN"
    elif (stock_name == "Google"):
        ticker_name = "GOOG"
    elif (stock_name == "Apple"):
        ticker_name = "AAPL"
    elif (stock_name == "Microsoft"):
        ticker_name = "MSFT"
    elif (stock_name == "Tesla"):
        ticker_name = "TSLA"


    elif (stock_name == "Bitcoin"):
        ticker_name = "BTC-USD"
    elif (stock_name == "Ethereum"):
        ticker_name = "ETH-USD"

    elif (stock_name == "Aixtron"):
        ticker_name = "AIXA.DE"
    elif (stock_name == "Gaia"):
        ticker_name = "GAIA"
    elif (stock_name == "SunOpta"):
        ticker_name = "STKL"
    elif (stock_name == "Infineon"):
        ticker_name = "IFX.DE"
    else:
        ticker_name = stock_name

    ticker = yf.Ticker(ticker_name)
    dataset = ticker.history(period='max').get(data_columns)
    dataset  = dataset[~(dataset == 0).any(axis=1)]
    dataset.isna().any()
    return dataset

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