import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
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


stock_names = ["Amazon","Google","Microsoft","Tesla","Bitcoin"]
stock_names = ["AAL"]
model_names = ["MixedTech_60_14"]*len(stock_names)
view_lengths = [60]*len(stock_names)
eval_perfs = [True]*len(stock_names)


for num in range(len(stock_names)):

        stock_name = stock_names[num]
        model_name = model_names[num]
        view_length_1 = 14
        view_length_2 = view_lengths[num]
        evaluate_performance = eval_perfs[num]


        batch_size = 32
        normalize = True
        window_size = 61
        data_columns = ['4. close','1. open', '5. volume']
        #data_columns = ['4. close']
        interval_min = 0
        interval_max = None
        show = False

        color_styles = ['#1f77b4','#2ca02c','#ff7f0e','#d62728','#9467bd','#8c564b','#bcbd22']

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
        else:
            print("Stock not available")
            quit()

        print(datasets[0].tail())

        data_test = [pd.DataFrame(datasets[0]).get(data_columns).values[interval_min:interval_max]]

        dataframe = pd.DataFrame(datasets[0])


        data_dates = pd.to_datetime(dataframe.index.values, format='%Y-%m-%d')
        # dates.apply(lambda x: x.strftime('%Y-%m-%d'))
        data_dates = [d.date() for d in data_dates]


        model = Model()
        model.load(model_name)

        x, y = model.init_window_data(data_test, window_size, False)
        real_stock_price = np.concatenate(y)

        relative_prediction_positions = [i * 7 for i in range(0,4)]  # [0,5,10,15,20,30,35,40]
        prediction_lengths = [7 for r in range(len(relative_prediction_positions))]+[30]
        relative_prediction_positions = relative_prediction_positions+[0]


        fig = plt.figure(figsize=(16,7))


        ax1 = plt.subplot2grid((3, 7), (0, 0), colspan=3, rowspan=3)
        ax2 = plt.subplot2grid((3, 7), (0, 3), colspan=3, rowspan=3)

        ax1.plot([data_dates[-1]], [real_stock_price[-1]], 'o', label="Heute", color=color_styles[1], zorder=10)
        ax2.plot([data_dates[-1]], [real_stock_price[-1]], 'o', label="Heute", color=color_styles[1], zorder=10)

        ax1.plot(data_dates[-view_length_1:], real_stock_price[-view_length_1:],'-', label ='bisheriger Preisverlauf',color=color_styles[0])
        ax2.plot(data_dates[-view_length_2:], real_stock_price[-view_length_2:],'-', label ='bisheriger Preisverlauf',color=color_styles[0])

        #2 week plot
        for p in range(2):
            pos = p
            prediction_length = 7
            predicted_stock_price = model.predict_sequence(x[-pos-1], window_size, normalize, prediction_length)
            predicted_stock_price.insert(0, y[-pos - 1][0])
            prediction_dates = [data_dates[-pos - 1]]
            for j in range(1, prediction_length + 1):
                prediction_dates.append(prediction_dates[j - 1] + dt.timedelta(days=1))
            ax1.plot(prediction_dates, predicted_stock_price, '--',
                     label=str(prediction_length) + '-Tage Vorhersage vor '+str(pos)+' Tagen',color=color_styles[1+p],zorder=5)
        zorder=5
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

            ax2.plot(prediction_dates, predicted_stock_price,
                     '--', label=str(prediction_length) + '-Tage Vorhersage vor ' + str(pos) + " Tagen",color = color_styles[1+i],zorder=zorder)
            zorder =0
            # Visualising the results
        fig.suptitle(stock_name, fontweight='bold',y=1.05,fontsize=20)
        ax1.set_title("letzte 2 Wochen")
        ax2.set_title("letzte " + str(view_length_2) + " Tage")
        ax1.set_xlabel('Datum')
        ax2.set_xlabel('Datum')
        ax1.set_ylabel('Preis / USD')
        ax2.set_ylabel('Preis / USD')
        ax1.legend(loc=0)
        ax2.legend(loc=0)

        props = dict(boxstyle='round,pad=1', facecolor=color_styles[0],edgecolor=color_styles[0], alpha=0.5)

        if evaluate_performance:
            prediction_sign_rates,prediction_errors = model.evaluate_prediction(x, y, 100, window_size, normalize, 7)
            print("prediction sign rate ",prediction_sign_rates)
            print("prediction errors ",prediction_errors)

            prediction_range = np.arange(7)
            errorinfo = "Abweichung: \n"

            for i in prediction_range:
                errorinfo += r"$\delta_" + str(i + 1) + "=" + \
                             '{0:.1f}'.format(prediction_errors[i] * 100.) + "\%$"
                if (i < len(prediction_sign_rates) - 1):
                    errorinfo += "\n"

            ax2.text(1.05, 0.8, errorinfo, transform=ax2.transAxes, fontsize=10,
                     verticalalignment='top', bbox=props)

        timestamp = dt.datetime.now().strftime("%d.%m.%Y / %H:%M:%S")
        ax2.text(1.05, 0.35, "Letzte Aktualisierung:\n" + timestamp, transform=ax2.transAxes, fontsize=10,
                 verticalalignment='top', bbox=props)

        ax2.text(1.05, 0.15, "Modell:\n" + model_name, transform=ax2.transAxes, fontsize=10,
                 verticalalignment='top', bbox=props)

        ax1.spines['top'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax2.spines['left'].set_visible(False)

        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.%Y'))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.%Y'))
        ax1.xaxis.set_major_locator(mdates.DayLocator(interval=int(view_length_2 / 7)))
        ax2.xaxis.set_major_locator(mdates.DayLocator(interval=int(view_length_2 / 7)))
        #plt.gcf().autofmt_xdate()
        plt.setp(plt.xticks()[1], rotation=30, ha='right')
        fig.tight_layout()
        ax1.grid(True,ls='--',lw=.5,c='k',alpha=.3)
        ax2.grid(True,ls='--',lw=.5,c='k',alpha=.3)

        abs_dir = os.path.dirname(os.path.realpath(__file__))
        plt.savefig(os.path.join(abs_dir, "figs/" + stock_name + "_" + str(view_length_2) + ".png"),bbox_inches='tight')
        if(show):
            plt.show()
        else:
            plt.close()
        print("Test for " + stock_name +" with model " + model_name + " for " + str(view_length_2) + " days done")
