# Importing the Keras libraries and packages

import numpy as np
from numpy import newaxis
from keras.layers import Dense, Activation, Dropout, LSTM,GRU
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint,LearningRateScheduler
from keras import optimizers
import keras
import keras.backend as K
import os
import math
import time


class Model:
    """A class for an building and inferencing an lstm model"""
    def __init__(self,name):
        self.name = name
        self.abs_dir = os.path.dirname(os.path.realpath(__file__))


    def build(self, configs):
        self.model = Sequential()

        for layer in configs['model']['layers']:
            neurons = layer['neurons'] if 'neurons' in layer else None
            dropout_rate = layer['dropout_rate'] if 'dropout_rate' in layer else None
            recurrent_dropout = layer['recurrent_dropout'] if 'recurrent_dropout' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            return_seq = layer['return_seq'] if 'return_seq' in layer else None
            input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
            input_dim = layer['input_dim'] if 'input_dim' in layer else None

            if layer['type'] == 'dense':
                self.model.add(Dense(neurons, activation=activation))
            if layer['type'] == 'lstm':
                self.model.add(LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq,dropout=dropout_rate,recurrent_dropout=recurrent_dropout))
            if layer['type'] == 'gru':
                self.model.add(GRU(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq,dropout=dropout_rate,recurrent_dropout=recurrent_dropout))
            if layer['type'] == 'dropout':
                self.model.add(Dropout(rate = dropout_rate))

        if configs['model']['optimizer']=='adam':
            optimizer = optimizers.Adam()
        elif configs['model']['optimizer']=='nadam':
            optimizer = optimizers.Nadam()
        elif configs['model']['optimizer'] == 'nesterov':
            optimizer = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        elif configs['model']['optimizer'] == 'rmsprop':
            optimizer = optimizers.RMSprop(lr=0.01, rho=0.9, epsilon=None, decay=0.0)
        else:
            optimizer = configs['model']['optimizer']

        #self.model.compile(loss=configs['model']['loss'], optimizer=optimizer)
        self.model.compile(loss=configs['model']['loss'], optimizer=optimizer)

        print('[Model] Model Compiled')



    '''***************************************TRAINING*****************************************************'''
    def train_generator(self, x_train, y_train, epochs, batch_size, steps_per_epoch, shuffle=True, early_stopping_patience=1000, x_val=None, y_val=None):

        training_batch_generator = BatchGenerator(x_train,y_train,batch_size,shuffle)

        early_stopping_callback = EarlyStopping(monitor='loss', patience=early_stopping_patience)

        loss_weight = 0.99

        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size, %s batches per epoch' % (epochs, batch_size, steps_per_epoch))
        time_start = time.time()
        if((x_val != None).all() and (y_val!= None).all()):
            validation_batch_generator = BatchGenerator(x_val, y_val, 10, shuffle)

            checkpoint = ModelCheckpoint(self.abs_dir + "/models/rnn_" + self.name + "_cp", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
            callbacks_list = [checkpoint, early_stopping_callback]

            self.model.fit_generator(
                training_batch_generator,
                validation_data=validation_batch_generator,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                workers=1,
                callbacks=callbacks_list
                #class_weight={0:loss_weight,1:1.-loss_weight}
            )
        else:
            checkpoint = ModelCheckpoint(self.abs_dir + "/models/rnn_" + self.name + "_cp", monitor='loss', verbose=1, save_best_only=True, mode='min')
            callbacks_list = [checkpoint, early_stopping_callback]

            self.model.fit_generator(
                training_batch_generator,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                workers=1,
                callbacks=callbacks_list
                #class_weight={0: loss_weight, 1: 1. - loss_weight}
            )

        print('[Model] Training completed in ' + '{0:.1f}'.format(time.time()-time_start) + "s")

    def step_decay_lr(self,epoch,current_lr):
        initial_lr = 0.1
        drop = 0.1
        epochs_drop = 2
        lr = initial_lr * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
        print("[Model] Learning rate set to "+str(lr))
        return lr

    '''***************************************PREDICTION*****************************************************'''
    def predict_point_by_point(self, data, window_size, normalize,n_outputs):
        # Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
        x_data,_,_ = self.window_data(data, window_size, False,n_outputs)
        x_norm,_,_  = self.window_data(data, window_size, True,n_outputs)
        print('[Model] Predicting Point-by-Point...')
        predicted = self.inverse_transform_prediction(x_data,[self.model.predict(x_norm)[0,0]])
        #predicted = np.reshape(predicted, (predicted.size,))
        return predicted

    def predict_sequence(self, data_initial, window_size,normalize, prediction_len, n_outputs):
        # print("mulit predict for " + str(prediction_length) + " steps and window size "+ str(window_size))
        curr_frame = data_initial
        prediction = []
        for i in range(prediction_len):
            #print("frame ",i,curr_frame)
            curr_frame_norm = self.relative_normalize_window(curr_frame, normalize)
            predicted = self.model.predict(curr_frame_norm)[0,0]
            prediction.append(self.inverse_transform_prediction([curr_frame], [predicted])[0])

            curr_frame = curr_frame[1:]
            curr_tmp = np.array(curr_frame[-1])
            curr_tmp[0] = prediction[-1]
            #print(curr_tmp)
            curr_frame = np.insert(curr_frame, window_size - n_outputs-1, curr_tmp, axis=0)
        return prediction

    def predict_sequences_multiple(self, data,window_size, normalize, prediction_len, n_outputs):
        # Predict sequence of 50 steps before shifting prediction run forward by 50 steps
        x_data,_,_  = self.window_data(data, window_size, False,n_outputs)
        print('[Model] Predicting Sequences Multiple...')
        prediction_seqs = []
        for i in range(int(np.shape(data)[1] / prediction_len)-1):
            curr_frame = x_data[i * prediction_len]
            prediction = []
            for j in range(prediction_len):
                #print(curr_frame)
                curr_frame_norm = self.relative_normalize_window(curr_frame, normalize)
                #print(curr_frame_norm)
                predicted = self.model.predict(curr_frame_norm)[0,0]
                prediction.append(self.inverse_transform_prediction([curr_frame],[predicted])[0])
                curr_frame = curr_frame[1:]
                curr_tmp = np.array(curr_frame[-1])
                curr_tmp[0] = prediction[-1]
                curr_frame = np.insert(curr_frame, window_size - n_outputs - 1, curr_tmp, axis=0)
            prediction_seqs.append(prediction)
        #print(prediction_seqs)
        return prediction_seqs

    def evaluate_prediction(self, dates,x, y, evaluation_length, window_size, normalize, prediction_length,n_outputs):
        reference_data = y[-evaluation_length-prediction_length:].flatten()
        reference_dates = dates[-evaluation_length-prediction_length:]
        prediction_sign_counter = np.zeros(prediction_length)
        prediction_rates = np.zeros(prediction_length)
        prediction_rates_sq = np.zeros(prediction_length)
        p_counter = np.zeros(prediction_length)
        for pos in range(evaluation_length):
            if(pos % 10 == 0):
                print("evaluating prediction at position "+str(pos))

            index = -pos-prediction_length-1

            date = reference_dates[index]
            data_initial = x[index]

            prediction = self.predict_sequence(data_initial, window_size,normalize,prediction_length,n_outputs)
            prediction = np.array(prediction).flatten()
            d_ref = np.zeros(prediction_length)
            d_pred = np.zeros(prediction_length)
            d_ratio = np.zeros(prediction_length)
            ones = np.zeros(prediction_length)


            delta_days = np.array([(reference_dates[index + p + 1]- date).days for p in range(prediction_length)])
            #print("delta days",delta_days)
            for p in range(prediction_length):
                ds = np.where(delta_days==p+1)[0]
                #print(ds)
                if(len(ds)>0):
                    #d is index of delta_days where it's equal to p+1, thus p+1 days from index means d+1 in dates
                    d = ds[0]
                    d_ref[p] = reference_data[index+d+1] - reference_data[index]
                    d_pred[p] = prediction[p] - reference_data[index]
                    d_ratio[p] = prediction[p]/reference_data[index+d+1]
                    ones[p] = 1
                    p_counter[p]+=1

            prediction_sign_counter += np.where(np.multiply(d_ref,d_pred)>0,1,0)
            prediction_rates += d_ratio-ones
            prediction_rates_sq += np.power(d_ratio-ones,2)

        prediction_sign_mean = prediction_sign_counter/p_counter
        prediction_mean = prediction_rates/p_counter
        prediction_mean_sq = prediction_rates_sq/p_counter
        #print(prediction_mean,prediction_mean_sq,p_counter)
        prediction_error = np.sqrt(prediction_mean_sq - np.power(prediction_mean,2)*(1-1/p_counter))
        #print(prediction_mean,prediction_error)
        return(prediction_sign_mean,prediction_mean,prediction_error)


    '''***************************************DATA OPERATIONS*****************************************************'''

    def window_data(self,data,window_size,normalize,n_outputs):
        data_x = []
        data_y = []
        series_y = []
        for d in data:
            for i in range(len(d) - window_size):
                x, y,s_y = self.get_next_window(d, i, window_size, normalize,n_outputs)
                data_x.append(x)
                data_y.append(y)
                series_y.append(s_y)
        print("[Model] Total data size is " + str(len(data_x)))
        return np.array(data_x),np.array(data_y),np.array(series_y)

    def inverse_transform_prediction(self, x_data, prediction_data):
        tranformed_predictions = []
        for i in range(len(prediction_data)):
            inputs = x_data[i]
            prediction = prediction_data[i]
            prediction_input = inputs[0]
            #print(prediction_input)
            #print(prediction)
            tranformed_predictions.append(prediction_input[0] * (1 + prediction))
        return (tranformed_predictions)

    def relative_normalize_window(self, window_data, single_window=False):
        '''Normalise window with a base value of zero'''
        normalised_data = []
        window_data = [window_data] if single_window else window_data
        for window in window_data:
            normalised_window = []
            for col_i in range(window.shape[1]):
                normalised_col = [((float(p) / float(window[0, col_i])) - 1) for p in window[:, col_i]]
                normalised_window.append(normalised_col)
            normalised_window = np.array(normalised_window).T  # reshape and transpose array back into original multidimensional format
            normalised_data.append(normalised_window)

        return np.array(normalised_data)

    def generate_sequential_batch(self,data_x,data_y,batch_size,shuffle):
        '''Yield a generator of training data from filename on given list of cols split for train/test'''
        data_length = np.shape(data_x)[0]
        window_size = np.shape(data_x)[1]
        indices = np.arange(data_length)
        #print("data x shape ",np.shape(self.data_x))
        if shuffle: np.random.shuffle(indices)
        i = 0
        while i < data_length:
            x_batch = []
            y_batch = []
            for b in range(batch_size):
                x = data_x[indices[i]]
                y = data_y[indices[i]]
                x_batch.append(x)
                y_batch.append(y)
                i += 1

                if i >= data_length:
                    # stop-condition for a smaller final batch if data doesn't divide evenly
                    #print()
                    #print(i,np.shape(x_batch),np.shape(y_batch))
                    yield np.array(x_batch), np.array(y_batch)
                    i = 0
                    if shuffle: np.random.shuffle(indices)

            yield np.array(x_batch), np.array(y_batch)

    def get_next_window(self, data, i, seq_len, normalize,n_outputs):
        '''Generates the next data window from the given index location i'''
        window = data[i:i + seq_len]
        window = self.relative_normalize_window(window, single_window=True)[0] if normalize else window
        x = window[:-n_outputs]
        y = window[-n_outputs:,0]
        s_y = window[-n_outputs, 0]
        return x, y, s_y

    '''***************************************SAVE / LOAD*****************************************************'''

    def save(self):
        self.model.save(self.abs_dir+"/models/rnn_" + self.name)
        print("[Model] " + self.name + " saved")

    def load(self):
        self.model = load_model(self.abs_dir+"/models/rnn_" + self.name)
        print("[Model] " + self.name + " loaded")

class BatchGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data_x,data_y,batch_size,shuffle):
        'Initialization'
        self.batch_size = batch_size
        self.data_x = data_x
        self.data_y = data_y
        self.data_len = len(data_y)
        self.indices = np.arange(self.data_len)
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        num_batches = int(np.floor(self.data_len / self.batch_size))
        return num_batches

    def __getitem__(self, index):
        'Generate one batch of data'

        x_batch = []
        y_batch = []
        for b in range(self.batch_size):
            x = self.data_x[self.indices[index]]
            y = self.data_y[self.indices[index]]
            x_batch.append(x)
            y_batch.append(y)

        return np.array(x_batch), np.array(y_batch)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.indices)