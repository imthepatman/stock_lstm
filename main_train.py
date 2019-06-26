import json
import sys
import pandas as pd
from data_ops import *
from model_fairseq import *
import math


if(len(sys.argv)==5) :

    stock_name = sys.argv[1]
    model_name = sys.argv[2]
    epochs = int(sys.argv[3])
    resume = sys.argv[4]
    n_outputs = 1
    window_size = 5 + n_outputs
    interval_min = -1*365
    interval_max = -100
    normalize = True

    batch_size = 1
    shuffle = True

    test_model = True


    abs_dir = os.path.dirname(os.path.realpath(__file__))
    config = json.load(open(abs_dir+'/model_config_fairseq.json', 'r'))

    model = Model(model_name)

    if (resume == "y"):
        model.build(config,load=True)
    elif (resume == "n"):
        model.build(config,load=False)
    else:
        print("resume has to be y/n")
        quit()

    data_columns = config["data_columns"]
    datasets = get_datasets(stock_name,data_columns)

    print(datasets[0].tail())

    filter_window_size = 21
    filter_order = 5

    data_original = [pd.DataFrame(ds).values for ds in datasets]
    data_train  = [filter_data(d[interval_min:interval_max],filter_window_size,filter_order) for d in data_original]

    x_train, y_train, _ = model.window_data(data_train, window_size, normalize,n_outputs)

    steps_per_epoch = math.ceil((len(x_train) / batch_size))

    if(interval_max!=None):

        data_val = [filter_data(d[interval_max:], filter_window_size, filter_order) for d in data_original]
        x_val, y_val,_ = model.window_data(data_val, window_size, normalize,n_outputs)

        model.train_generator(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            shuffle=shuffle,
            x_val=x_val,
            y_val=y_val
        )
    else:
        model.train_generator(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            shuffle=shuffle
        )
    model.save()

    if(test_model):
        data_test = [data_original[0][-1000:]]

        _,_, s_y_test = model.window_data(data_test, window_size, False,n_outputs)

        predictions_multiseq = model.predict_sequences_multiple(data_test, window_size, normalize, window_size,n_outputs)
        #predictions_fullseq = model.predict_sequence_full(data_test, window_size,normalize)
        #predictions_pointbypoint = model.predict_point_by_point(data_test, window_size, normalize)

        plot_results_multiple(predictions_multiseq, s_y_test, window_size)
        #plot_results(predictions_fullseq, y_test)
        #plot_results(predictions_pointbypoint, y_test)

else:
    print("Wrong number of arguments. Exiting.")
    quit()
