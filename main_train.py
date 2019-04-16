import json
import sys
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.cryptocurrencies import CryptoCurrencies
import pandas as pd
import math
from model import *

if(len(sys.argv)==5) :

    stock_name = sys.argv[1]
    model_name = sys.argv[2]
    epochs = int(sys.argv[3])
    resume = sys.argv[4]

    data_columns = ['4. close','1. open', '5. volume']
    window_size = 31
    interval_min = 0
    interval_max = None
    normalize = True

    batch_size = 32
    steps_per_epoch = 100
    shuffle = True

    test_model = False

    if (stock_name == "Amazon"):
        ts = TimeSeries(key='PJNUY6BXU7LQI3P1', output_format='pandas', indexing_type='date')
        dataset, meta_data = ts.get_daily(symbol='AMZN', outputsize='full')
        dataset.isna().any()
        datasets = [dataset]
    elif (stock_name == "Google"):
        ts = TimeSeries(key='PJNUY6BXU7LQI3P1', output_format='pandas', indexing_type='date')
        dataset, meta_data = ts.get_daily(symbol='GOOG', outputsize='full')
        datasets = [dataset]
    elif (stock_name == "Apple"):
        ts = TimeSeries(key='PJNUY6BXU7LQI3P1', output_format='pandas', indexing_type='date')
        dataset, meta_data = ts.get_daily(symbol='AAPL', outputsize='full')
        datasets = [dataset]
    elif (stock_name == "Microsoft"):
        ts = TimeSeries(key='PJNUY6BXU7LQI3P1', output_format='pandas', indexing_type='date')
        dataset, meta_data = ts.get_daily(symbol='MSFT', outputsize='full')
        datasets = [dataset]
    elif (stock_name == "AAL"):
        ts = TimeSeries(key='PJNUY6BXU7LQI3P1', output_format='pandas', indexing_type='date')
        dataset, meta_data = ts.get_daily(symbol='AAL', outputsize='full')
        dataset.isna().any()
        datasets = [dataset]
    elif (stock_name == "Tesla"):
        ts = TimeSeries(key='PJNUY6BXU7LQI3P1', output_format='pandas', indexing_type='date')
        dataset, meta_data = ts.get_daily(symbol='TSLA', outputsize='full')
        datasets = [dataset]
    elif (stock_name == "Bitcoin"):
        cc = CryptoCurrencies(key='PJNUY6BXU7LQI3P1', output_format='pandas', indexing_type='date')
        dataset, meta_data = cc.get_digital_currency_daily(symbol='BTC', market='USD')
        dataset.isna().any()
        datasets = [dataset]
        data_columns = ["4a. close (USD)", "1a. open (USD)", "5. volume"]

    elif (stock_name== "MixedTech"):
        symbols = ["GOOG","AMZN","MSFT"]
        datasets = []
        for s in symbols:
            ts = TimeSeries(key='PJNUY6BXU7LQI3P1', output_format='pandas', indexing_type='date')
            dataset, meta_data = ts.get_daily(symbol=s, outputsize='full')
            dataset.isna().any()
            datasets.append(dataset)
    else:
        print("Stock not available")
        quit()

    print(datasets[0].tail())

    data_train  = [pd.DataFrame(ds).get(data_columns).values[interval_min:interval_max] for ds in datasets]
    #data_train = [np.reshape(np.abs(np.sinc(np.linspace(-10,10,1000)))+1,(-1,1))]
    abs_dir = os.path.dirname(os.path.realpath(__file__))

    config = json.load(open(abs_dir+'/model_config.json', 'r'))
    model = lstm()

    data_x, datay = model.init_window_data(data_train,window_size,normalize)

    if (resume == "y"):
        model.load(model_name)
    elif (resume == "n"):
        model.build(config)
    else:
        print("resume has to be y/n")
        quit()

    #model.train(data_train, window_size, normalize, epochs, batch_size, validation_split)

    steps_per_epoch = math.ceil((len(data_x)/ batch_size))
    model.train_generator(
        epochs=epochs,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
        shuffle=shuffle
    )
    model.save(model_name)

    if(test_model):
        data_test = [pd.DataFrame(datasets[0]).get(data_columns).values[-1000:None]]
        data_test = data_train

        x_test,y_test = model.init_window_data(data_test, window_size, False)

        predictions_multiseq = model.predict_sequences_multiple(data_test, window_size, normalize, window_size)
        #predictions_fullseq = model.predict_sequence_full(data_test, window_size,normalize)
        predictions_pointbypoint = model.predict_point_by_point(data_test, window_size, normalize)

        #print(np.shape(y_test))
        #print(np.shape(predictions_multiseq))
        plot_results_multiple(predictions_multiseq, y_test, window_size)
        #plot_results(predictions_fullseq, y_test)
        plot_results(predictions_pointbypoint, y_test)

else:
    print("Wrong number of arguments. Exiting.")
    quit()