import matplotlib
matplotlib.use('Agg')
import pandas as pd
from model import *
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import datetime as dt
from data_ops import *
import os
import json
#plt.rcParams["font.family"] = "monospace"
matplotlib.rc('font',**{'family':'monospace'})
matplotlib.rcParams['mathtext.fontset'] = 'dejavusans'


stock_names=["Infineon","Aixtron","Gaia","SunOpta","Bitcoin"]#
#model_names =["portfolio_2_256-256_10y"]# ["portfolio_2_256-256_10y"]*3 +
#model_names = ["portfolio_2_256-256_5y"]*4 + ["Bitcoin_2_256-256_5y"]

stock_names = ["SunOpta"]
model_names = ["SunOpta_2_20_5y_filter-5","Aixtron_1_10_5y_filter-3","Gaia_1_10_5y_filter-3","SunOpta_1_10_5y_filter-3","Bitcoin_1_10_5y_filter-3"]

#model_names = ["Aixtron_1_256-256_5y","Gaia_1_256-256_5y","SunOpta_1_256-256_5y"]
view_lengths = [60]*len(stock_names)
eval_perfs = [True]*len(stock_names)


for num in range(len(stock_names)):

        stock_name = stock_names[num]
        model_name = model_names[num]
        view_length_1 = 14
        view_length_2 = view_lengths[num]
        evaluate_performance = eval_perfs[num]


        normalize = True
        window_size = 61
        #data_columns = ['4. close']
        interval_min = 0
        interval_max = None
        show = False

        color_lines = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd', '#8c564b', '#bcbd22']
        color_palette = {"blue":"#1f77b4","wine_red":"#794044","green":"#2ca02c"}

        abs_dir = os.path.dirname(os.path.realpath(__file__))
        config = json.load(open(abs_dir + '/model_config.json', 'r'))
        data_columns = config["data_columns"]
        # data_columns = ["Close","Open","High","Low","Volume"]
        # data_columns = ["Close","Open","Volume"]

        datasets = get_datasets(stock_name,data_columns,append_intraday=True)
        print(datasets[0].tail())

        filter_window_size = 21
        filter_order = 5

        data_original = [pd.DataFrame(ds).values[interval_min:interval_max] for ds in datasets]
        # data_original = [np.reshape(np.sin(4*np.linspace(-10,10,1000))+2,(-1,1))]
        data_test = [filter_data(d, filter_window_size, filter_order) for d in data_original]

        dataframe = pd.DataFrame(datasets[0])


        data_dates = pd.to_datetime(dataframe.index.values, format='%Y-%m-%d')
        # dates.apply(lambda x: x.strftime('%Y-%m-%d'))
        data_dates = [d.date() for d in data_dates]


        model = Model(model_name)
        model.load()

        x_ori, y_ori = model.window_data(data_original, window_size, False)
        x_test, y_test = model.window_data(data_test, window_size, False)

        filtered_stock_price = np.concatenate(y_test)
        real_stock_price = np.concatenate(y_ori)


        fig = plt.figure(figsize=(16,7))


        ax1 = plt.subplot2grid((3, 6), (0, 0), colspan=3, rowspan=3)
        ax2 = plt.subplot2grid((3, 6), (0, 3), colspan=3, rowspan=3)

        today_date = data_dates[-1].strftime("%d.%m.%Y")
        ax1.plot([data_dates[-1]], [real_stock_price[-1]], 'o', label="Heute - "+today_date, color=color_palette["green"], zorder=10)
        ax2.plot([data_dates[-1]], [real_stock_price[-1]], 'o', label="Heute - "+today_date, color=color_palette["green"], zorder=10)

        ax1.plot(data_dates[-view_length_1:], real_stock_price[-view_length_1:],'-', label ='bisheriger Preisverlauf',color=color_palette["blue"])
        ax1.plot(data_dates[-view_length_1:], filtered_stock_price[-view_length_1:],':', label ='gefilterter Preisverlauf',color=color_palette["blue"],alpha=0.7)
        ax2.plot(data_dates[-view_length_2:], real_stock_price[-view_length_2:],'-', label ='bisheriger Preisverlauf',color=color_palette["blue"])
        ax2.plot(data_dates[-view_length_2:], filtered_stock_price[-view_length_2:],':', label ='gefilterter Preisverlauf',color=color_palette["blue"],alpha=0.7)

        main_prediction = []
        #2 week plot
        for p in range(2):
            pos = p
            prediction_length = 7
            predicted_stock_price = model.predict_sequence(x_test[-pos - 1], window_size, normalize, prediction_length)
            predicted_stock_price.insert(0, y_ori[-pos - 1][0])
            prediction_dates = [data_dates[-pos - 1]]
            for j in range(1, prediction_length + 1):
                prediction_dates.append(prediction_dates[j - 1] + dt.timedelta(days=1))
            delta_days = (data_dates[- 1] - data_dates[-pos - 1]).days

            #for later error tube
            if(p==0):
                main_prediction = [prediction_dates,np.array(predicted_stock_price)]
            ax1.plot(prediction_dates, predicted_stock_price, '--',
                     label=str(prediction_length) + '-Tage Vorhersage vor '+str(delta_days)+' Tagen', color=color_lines[1 + p], zorder=5)
        zorder=5

        relative_prediction_positions = [i * 7 for i in range(0, 4)]  # [0,5,10,15,20,30,35,40]
        prediction_lengths = [7 for r in range(len(relative_prediction_positions))] + [30]
        relative_prediction_positions = relative_prediction_positions + [0]

        for i in range(len(relative_prediction_positions)):
            pos = relative_prediction_positions[i]
            prediction_length = prediction_lengths[i]

            print("predicting "+ str(prediction_length) +" days from " + str(pos) + " days ago")

            # Getting the predicted stock price

            #predict test by looking at last window_size entries of dataset_total
            data_initial = x_test[-pos - 1]

            predicted_stock_price = model.predict_sequence(data_initial, window_size,normalize, prediction_length)
            predicted_stock_price.insert(0, y_ori[-pos - 1][0])

            #predicted_stock_price = sc.inverse_transform(np.reshape(predicted_stock_price,(-1,1)))

            prediction_dates = [data_dates[-pos-1]]
            for j in range(1,prediction_length+1):
                prediction_dates.append(prediction_dates[j-1] + dt.timedelta(days=1))

            #print(real_stock_price[-1],predicted_stock_price)

            delta_days = (data_dates[-1] - data_dates[-pos - 1]).days
            ax2.plot(prediction_dates, predicted_stock_price,
                     '--', label=str(prediction_length) + '-Tage Vorhersage vor ' + str(delta_days) + " Tagen", color = color_lines[1 + i], zorder=zorder)
            zorder =0
            # Visualising the results
        fig.suptitle(stock_name, fontweight='bold',y=1.05,fontsize=20)
        ax1.set_title("letzte 2 Wochen",fontsize=14)
        ax2.set_title("letzte " + str(view_length_2) + " Tage",fontsize=14)
        ax1.set_xlabel('Datum')
        ax2.set_xlabel('Datum')
        ax1.set_ylabel('Preis')
        ax2.set_ylabel('Preis')
        ax1.legend(loc=0)
        ax2.legend(loc=0)

        props = dict(boxstyle='round,pad=1', facecolor=color_palette["blue"],edgecolor=color_palette["blue"], alpha=0.5)

        if evaluate_performance:
            prediction_sign_rates, prediction_mean, prediction_error = model.evaluate_prediction(data_dates, x_test, y_ori, 100, window_size, normalize, 7)
            print("prediction sign rate ",prediction_sign_rates)
            print("prediction mean ",prediction_mean)
            print("prediction errors ", prediction_error)

            prediction_range = np.arange(7)
            errorinfo = "Vorhersage Info: \n"

            #col_labels = ["Tage", "Trend", r"$\pm$",r"$\updownarrow$"]
            #row_labels = [str(i+1) for i in prediction_range]
            #plt.rc('text', usetex=True)
            #table = r"\begin{tabular}{ c | c | c | c } "+ col_labels[0] + r" & " + col_labels[1] + r" & " + col_labels[2] + r" & " + col_labels[3] + r" \\\hline "

            for i in prediction_range:
                #table+= row_labels[i] + " & {0:.1f}".format(prediction_mean[i] * 100.)+ " & {0:.1f}".format(prediction_error[i] * 100.)+ " & {0:.1f}".format(prediction_sign_rates[i] * 100.)
                errorinfo += r"$\delta_" + str(i + 1) + "=" + "{0:.1f}".format(prediction_mean[i] * 100.) + \
                            r"\pm" + "{0:.1f}".format(prediction_error[i] * 100.) + "\% \sim " + "{0:.1f}".format(prediction_sign_rates[i]* 100.) + "\%$"
                if (i < len(prediction_sign_rates) - 1):
                    #table+=r" \\\hline "
                    errorinfo += "\n"

            #table += r" \end{tabular}"
            #table = r'''\begin{tabular}{ c | c | c | c } & col1 & col2 & col3 \\\hline row1 & 11 & 12 & 13 \\\hline row2 & 21 & 22 & 23 \\\hline  row3 & 31 & 32 & 33 \end{tabular}'''
            ax2.text(1.05, 0.5, errorinfo, transform=ax2.transAxes, fontsize=12,
                     verticalalignment='top', bbox=props)
            #plt.rc('text', usetex=False)
            prediction_error = np.insert(prediction_error,0,0)
            ax1.fill_between(main_prediction[0],main_prediction[1]*(1-3*prediction_error),main_prediction[1]*(1+3*prediction_error),color=color_palette["green"], alpha=.1)
            ax1.fill_between(main_prediction[0],main_prediction[1]*(1-2*prediction_error),main_prediction[1]*(1+2*prediction_error),color=color_palette["green"], alpha=.2)
            ax1.fill_between(main_prediction[0],main_prediction[1]*(1-prediction_error),main_prediction[1]*(1+prediction_error),color=color_palette["green"], alpha=.3)
            ax2.fill_between(main_prediction[0], main_prediction[1] * (1 - 3 * prediction_error),
                             main_prediction[1] * (1 + 3 * prediction_error), color=color_palette["green"], alpha=.1)
            ax2.fill_between(main_prediction[0], main_prediction[1] * (1 - 2 * prediction_error),
                             main_prediction[1] * (1 + 2 * prediction_error), color=color_palette["green"], alpha=.2)
            ax2.fill_between(main_prediction[0], main_prediction[1] * (1 - prediction_error),
                             main_prediction[1] * (1 + prediction_error), color=color_palette["green"], alpha=.3)

        ax2.text(1.05, 0.08, "Modell: \n" + model_name, transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)

        timestamp = dt.datetime.now().strftime("%d.%m.%Y / %H:%M:%S")
        ax2.text(1.05, -0.05, "Letzte Aktualisierung:\n" + timestamp, transform=ax2.transAxes, fontsize=10,
                 verticalalignment='top', bbox=props)

        ax1.spines['top'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax2.spines['left'].set_visible(False)

        ax1.margins(0)
        ax2.margins(0)
        ax1.set_ylim(ax1.get_ylim()[0], ax1.get_ylim()[1] * 1.02)
        ax2.set_ylim(ax2.get_ylim()[0], ax2.get_ylim()[1] * 1.02)

        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.%Y'))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.%Y'))
        ax1.xaxis.set_major_locator(mdates.DayLocator(interval=int(view_length_2 / 7)))
        ax1.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
        ax2.xaxis.set_major_locator(mdates.DayLocator(interval=int(view_length_2 / 7)))
        ax2.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
        #plt.gcf().autofmt_xdate()
        plt.setp(plt.xticks()[1], rotation=30, ha='right')
        fig.tight_layout()
        ax1.grid(True,'major',ls='--',lw=.8,c='black',alpha=.3)
        ax1.grid(True,'minor',ls=':',lw=.5,c='k',alpha=.3)
        ax2.grid(True,'major',ls='--',lw=.8,c='black',alpha=.3)
        ax2.grid(True,'minor',ls=':',lw=.5,c='k',alpha=.3)

        ax1.fill_between(data_dates[-view_length_1:], ax1.get_ylim()[0], real_stock_price[-view_length_1:],
                         color=color_palette["blue"], alpha=.3)
        ax2.fill_between(data_dates[-view_length_2:], ax2.get_ylim()[0], real_stock_price[-view_length_2:],
                         color=color_palette["blue"], alpha=.3)


        plt.savefig(os.path.join(abs_dir, "figs/" + stock_name + "_" + str(view_length_2) + ".png"),bbox_inches='tight',dpi=100)
        if(show):
            plt.show()
        else:
            plt.close()
        print("Test for " + stock_name +" with model " + model_name + " for " + str(view_length_2) + " days done")