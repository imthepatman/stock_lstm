import json
import sys
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.cryptocurrencies import CryptoCurrencies
import pandas as pd
from model import *

if(len(sys.argv)==5) :

    stock_name = sys.argv[1]
    model_name = sys.argv[2]
    epochs = int(sys.argv[3])
    resume = sys.argv[4]

    batch_size = 32
    steps_per_epoch = 100
    normalize = True
    window_size = 60
    data_columns = ['4. close']
    interval_min = 0
    interval_max = None
    validation_split = 0.1
    test_model = False

    if (stock_name == "Amazon"):
        ts = TimeSeries(key='PJNUY6BXU7LQI3P1', output_format='pandas', indexing_type='date')
        dataset, meta_data = ts.get_daily(symbol='AMZN', outputsize='full')
    elif (stock_name == "Tesla"):
        ts = TimeSeries(key='PJNUY6BXU7LQI3P1', output_format='pandas', indexing_type='date')
        dataset, meta_data = ts.get_daily(symbol='TSLA', outputsize='full')
    elif (stock_name == "AAL"):
        ts = TimeSeries(key='PJNUY6BXU7LQI3P1', output_format='pandas', indexing_type='date')
        dataset, meta_data = ts.get_daily(symbol='AAL', outputsize='full')
    elif (stock_name == "Bitcoin"):
        cc = CryptoCurrencies(key='PJNUY6BXU7LQI3P1', output_format='pandas', indexing_type='date')
        dataset, meta_data = cc.get_digital_currency_daily(symbol='BTC', market='USD')
        data_columns = ["4a. close (USD)"]
    else:
        print("Stock not available")
        quit()

    print(dataset.tail())

    dataset.isna().any()
    dataframe = pd.DataFrame(dataset)
    data_train  = dataframe.get(data_columns).values[interval_min:interval_max]

    config = json.load(open('model_config.json', 'r'))
    model = lstm()

    if (resume == "y"):
        model.load(model_name)
    elif (resume == "n"):
        model.build(config)
    else:
        print("resume has to be y/n")
        quit()

    #model.train(data_train, window_size, normalize, epochs, batch_size, validation_split)
    #steps_per_epoch = math.ceil((len(x) -window_size) / batch_size)
    model.train_generator(
        data_train,window_size,normalize,
        epochs=epochs,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch
    )
    model.save(model_name)

    if(test_model):
        data_test = dataframe.get(data_columns).values[-1000:None]
        x_test,y_test = model.get_data(data_test, window_size, False)

        #predictions_multiseq = model.predict_sequences_multiple(data_test, window_size, normalize, window_size)
        #predictions_fullseq = model.predict_sequence_full(data_test, window_size,normalize)
        predictions_pointbypoint = model.predict_point_by_point(data_test, window_size, normalize)

        #print(np.shape(y_test))
        #print(np.shape(predictions_multiseq))
        #plot_results_multiple(predictions_multiseq, y_test, window_size)
        #plot_results(predictions_fullseq, y_test)
        plot_results(predictions_pointbypoint, y_test)