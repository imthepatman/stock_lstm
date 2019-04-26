import json
import sys
import pandas as pd
from data_ops import *
from model import *

if(len(sys.argv)==5) :

    stock_name = sys.argv[1]
    model_name = sys.argv[2]
    epochs = int(sys.argv[3])
    resume = sys.argv[4]
    window_size = 61
    interval_min = -10*365
    interval_max = -100
    normalize = True

    batch_size = 64
    shuffle = True

    test_model = False

    abs_dir = os.path.dirname(os.path.realpath(__file__))
    config = json.load(open(abs_dir+'/model_config.json', 'r'))

    datasets = get_datasets(stock_name)

    print(datasets[0].tail())

    data_columns = config["data_columns"]

    data_train  = [pd.DataFrame(ds).get(data_columns).values[interval_min:interval_max] for ds in datasets]
    #data_train = [np.reshape(np.abs(np.sin(4*np.linspace(-10,10,10000)))+1,(-1,1))]
    model = Model(model_name)

    data_x, datay = model.init_window_data(data_train,window_size,normalize)

    if (resume == "y"):
        model.load()
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
    model.save()

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
