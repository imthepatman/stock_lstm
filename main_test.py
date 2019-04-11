import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from model import *
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.cryptocurrencies import CryptoCurrencies
register_matplotlib_converters()
import datetime as dt
import sys
import os
matplotlib.rcParams['font.sans-serif']

if(len(sys.argv)==3) :

    stock_name = sys.argv[1]
    model_name = sys.argv[2]

    batch_size = 32
    normalize = True
    window_size = 60
    #sequence_length = 60
    data_columns = ['4. close']
    #data_columns = ['4. close', '5. volume']
    interval_min = -200
    interval_max = None
    view_length = 60
    show = False
    evaluate_performance = True

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

    #data_test = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
    dataset.isna().any()
    dataframe = pd.DataFrame(dataset)

    data_test = dataframe.get(data_columns).values[interval_min:interval_max]


    data_dates = pd.to_datetime(dataframe.index.values, format='%Y-%m-%d')
    # dates.apply(lambda x: x.strftime('%Y-%m-%d'))
    data_dates = [d.date() for d in data_dates]

    relative_prediction_positions = [i*7 for i in range(4)]#[0,5,10,15,20,30,35,40]
    prediction_lengths = [30] + [7 for r in range(len(relative_prediction_positions))]
    relative_prediction_positions = [0] + relative_prediction_positions


    model = lstm()
    model.load(model_name)

    x, y = model.get_data(data_test, window_size, False)
    real_stock_price = np.concatenate(y)



    fig = plt.figure(figsize=(8,6))
    gridspec.GridSpec(4, 3)
    ax = plt.subplot2grid((3,4), (0,0), colspan=3, rowspan=3)
    ax.plot(data_dates[-view_length:], real_stock_price[-view_length:], label ='Aktueller Preis')

    for i in range(len(relative_prediction_positions)):
        pos = relative_prediction_positions[i]
        prediction_length = prediction_lengths[i]

        print("predicting "+ str(prediction_length) +" days from " + str(pos) + " days ago")

        # Getting the predicted stock price

        #predict test by looking at last window_size entries of dataset_total
        data_initial = x[-pos-1]

        predicted_stock_price = model.predict_sequence(data_initial, window_size,normalize, prediction_length)
        predicted_stock_price.insert(0,y[-pos-1][0])

        #predicted_stock_price = sc.inverse_transform(np.reshape(predicted_stock_price,(-1,1)))

        prediction_dates = [data_dates[-pos-1]]
        for j in range(1,prediction_length+1):
            prediction_dates.append(prediction_dates[j-1] + dt.timedelta(days=1))

        #print(real_stock_price[-1],predicted_stock_price)
        ax.plot(prediction_dates, predicted_stock_price,'--', label=str(prediction_length) + '-Tage Vorhersage vor ' + str(pos) + " Tagen")

        # Visualising the results

    ax.plot([data_dates[-1]],[real_stock_price[-1]],'o',label="Heute")

    ax.set_title(stock_name + ' Preisentwicklung', fontweight='bold')
    ax.set_xlabel('Datum')
    ax.set_ylabel('Preis / USD')
    ax.legend(loc=0)

    props = dict(boxstyle='round,pad=1', facecolor='#3c98ad',edgecolor='#3c98ad', alpha=0.5)

    if evaluate_performance:
        prediction_sign_rates = model.evaluate_prediction_sign(x, y, 100, window_size,normalize, 7)
        print("prediction sign rate ",prediction_sign_rates)

        infotext = "Infos: \n"
        for i in range(len(prediction_sign_rates)):
            infotext += "$r_" + str(i + 1) + "=" + '{0:.0f}'.format(prediction_sign_rates[i] * 100.) + "\%$"
            if (i < len(prediction_sign_rates) - 1):
                infotext += "\n"

        ax.text(1.05, 0.9, infotext, transform=ax.transAxes, fontsize=13,
            verticalalignment='top', bbox=props)

    timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ax.text(1.05, 0.35, "Letzte Aktualisierung:\n" +timestamp, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)

    ax.text(1.05, 0.15, "Modell:\n" + model_name, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=10))
    plt.gcf().autofmt_xdate()
    fig.tight_layout()
    abs_dir = os.path.dirname(os.path.realpath(__file__))
    plt.savefig(os.path.join(abs_dir, "figs/" + stock_name + ".png"))
    if(show):
        plt.show()
    else:
        plt.close()

else:
    print("Stock name needed")
    quit()