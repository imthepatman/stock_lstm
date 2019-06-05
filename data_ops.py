import fix_yahoo_finance as yf
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import datetime as dt
import pandas as pd

def get_datasets(stock_name,data_columns,append_intraday=False):

    if (stock_name== "MixedTech"):
        symbols = ["Amazon","Tesla","Apple","Google","Microsoft"]
        datasets = []
        for s in symbols:
            datasets.append(get_single_dataset(s,data_columns,append_intraday))
        return datasets

    elif (stock_name== "crypto"):
        symbols = ["Bitcoin","Ethereum"]
        datasets = []
        for s in symbols:
            datasets.append(get_single_dataset(s,data_columns,append_intraday))
        return datasets

    elif (stock_name== "eko"):
        symbols = ["Aixtron","GAIA","SunOpta"]
        datasets = []
        for s in symbols:
            datasets.append(get_single_dataset(s,data_columns,append_intraday))
        return datasets

    elif (stock_name== "portfolio"):
        symbols = ["Infineon","Aixtron","GAIA","SunOpta"]
        datasets = []
        for s in symbols:
            datasets.append(get_single_dataset(s,data_columns,append_intraday))
        return datasets
    elif(stock_name=="SemiCon"):
        symbols = ["Aixtron", "Infineon"]
        datasets = []
        for s in symbols:
            datasets.append(get_single_dataset(s,data_columns,append_intraday))
        return datasets

    elif (stock_name== "SinTest"):
        return [pd.DataFrame(np.sin(4.*np.linspace(-10,10,1000))+2.,columns=['Close'])]
    elif (stock_name== "ArangeTest"):
        return [pd.DataFrame(np.arange(0,1000),columns=['Close'])]

    else:
        return [get_single_dataset(stock_name,data_columns,append_intraday)]



def get_single_dataset(stock_name,data_columns,append_intraday=False):
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

    #ticker = yf.Ticker(ticker_name)
    dataset = yf.download(ticker_name,period='max').get(data_columns)

    dataset  = dataset[~(dataset == 0).any(axis=1)]
    dataset.isna().any()

    if append_intraday:
        today = dt.datetime.today()
        dataset_intraday = yf.download(ticker_name,period='1d',interval='1m').get(data_columns)
        dataset_intraday  = dataset_intraday[~(dataset_intraday == 0).any(axis=1)]
        dataset_intraday.isna().any()

        mask = pd.to_datetime(dataset.index.values).strftime("%Y-%m-%d") == today.strftime("%Y-%m-%d")
        mask_intraday = pd.to_datetime(dataset_intraday.index.values).strftime("%Y-%m-%d") == today.strftime("%Y-%m-%d")
        if(not mask.any() and mask_intraday.any()):
            append_values = np.mean(dataset.values[-7:],axis=0)
            dataset_tmp = dataset_intraday[mask_intraday]
            append_values[0] = np.mean(dataset_tmp.values[-60:],axis=0)[0]
            data_tmp = pd.DataFrame([append_values],index=[today],columns = data_columns)
            dataset = pd.concat([dataset,data_tmp])

    return dataset

def filter_data(data_series,window_size=5,order=3):
    filtered_data = signal.savgol_filter(np.array(data_series),window_size,order,axis=0)
    return(filtered_data)
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