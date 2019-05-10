# Importing the Keras libraries and packages

import numpy as np
from numpy import newaxis
from keras.layers import Dense, Activation, Dropout, LSTM,GRU
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint,LearningRateScheduler
from keras import optimizers
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
            activation = layer['activation'] if 'activation' in layer else None
            return_seq = layer['return_seq'] if 'return_seq' in layer else None
            input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
            input_dim = layer['input_dim'] if 'input_dim' in layer else None

            if layer['type'] == 'dense':
                self.model.add(Dense(neurons, activation=activation))
            if layer['type'] == 'lstm':
                self.model.add(LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq,recurrent_dropout=dropout_rate))
            if layer['type'] == 'gru':
                self.model.add(GRU(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq,recurrent_dropout=dropout_rate))
            if layer['type'] == 'dropout':
                self.model.add(Dropout(rate = dropout_rate))

        if configs['model']['optimizer']=='adam':
            optimizer = optimizers.Adam()
        elif configs['model']['optimizer']=='nadam':
            optimizer = optimizers.Nadam()
        elif configs['model']['optimizer'] == 'nesterov':
            optimizer = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        elif configs['model']['optimizer'] == 'rmsprop':
            optimizer = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
        else:
            optimizer = configs['model']['optimizer']

        self.model.compile(loss=configs['model']['loss'], optimizer=optimizer)

        print('[Model] Model Compiled')



    '''***************************************TRAINING*****************************************************'''
    def train(self, data,sequence_length, normalize, epochs, batch_size,validation_split):
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size' % (epochs, batch_size))
        x,y = self.get_data(data,sequence_length,normalize)
        self.model.fit(
            x,
            y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split
        )

    def train_generator(self, epochs, batch_size, steps_per_epoch, shuffle=True, early_stopping_patience=1000):

        batch_generator = self.generate_sequential_batch

        early_stopping_callback = EarlyStopping(monitor='loss', patience=early_stopping_patience)
        learning_rate_scheduler_callback = LearningRateScheduler(self.step_decay_lr)
        checkpoint = ModelCheckpoint(self.abs_dir+"/models/rnn_"+self.name+"_cp", monitor='loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint,early_stopping_callback]

        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size, %s batches per epoch' % (epochs, batch_size, steps_per_epoch))
        time_start = time.time()
        self.model.fit_generator(
            batch_generator(batch_size,shuffle),
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            workers=1,
            callbacks=callbacks_list
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
    def predict_point_by_point(self, data, window_size, normalize):
        # Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
        x_data,y_data = self.init_window_data(data, window_size, False)
        x_norm, y_norm = self.init_window_data(data, window_size, True)
        print('[Model] Predicting Point-by-Point...')
        predicted = self.inverse_transform_prediction(x_data,self.model.predict(x_norm))
        #predicted = np.reshape(predicted, (predicted.size,))
        return predicted

    def predict_sequence(self, data_initial, window_size,normalize, prediction_len):
        # print("mulit predict for " + str(prediction_length) + " steps and window size "+ str(window_size))
        curr_frame = data_initial
        prediction = []
        for i in range(prediction_len):
            #print("frame ",i,curr_frame)
            curr_frame_norm = self.normalize_windows(curr_frame, normalize)
            predicted = self.model.predict(curr_frame_norm)
            prediction.append(self.inverse_transform_prediction([curr_frame], predicted)[0])

            curr_frame = curr_frame[1:]
            #curr_frame = np.insert(curr_frame, [window_size - 2], prediction[-1], axis=0)
            curr_frame = np.insert(curr_frame, [window_size - 2], [prediction[-1],curr_frame[-1][1]], axis=0)
        return prediction

    def predict_sequences_multiple(self, data,window_size, normalize, prediction_len):
        # Predict sequence of 50 steps before shifting prediction run forward by 50 steps
        x_data, y_data = self.init_window_data(data, window_size, False)
        print('[Model] Predicting Sequences Multiple...')
        prediction_seqs = []
        for i in range(int(np.shape(data)[1] / prediction_len)-1):
            curr_frame = x_data[i * prediction_len]
            prediction = []
            for j in range(prediction_len):
                #print(curr_frame)
                curr_frame_norm = self.normalize_windows(curr_frame,normalize)
                #print(curr_frame_norm)
                predicted = self.model.predict(curr_frame_norm)
                #print(predicted)
                prediction.append(self.inverse_transform_prediction([curr_frame],predicted)[0])
                curr_frame = curr_frame[1:]
                curr_frame = np.insert(curr_frame, [window_size - 2], prediction[-1], axis=0)
            prediction_seqs.append(prediction)
        #print(prediction_seqs)
        return prediction_seqs

    def predict_sequence_full(self, data, window_size):
        # Shift the window by 1 new prediction each time, re-run predictions on new window
        print('[Model] Predicting Sequences Full...')
        curr_frame = data[0]
        predicted = []
        for i in range(len(data)):
            predicted.append(self.model.predict(curr_frame[newaxis, :, :])[0, 0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size - 2], predicted[-1], axis=0)
        return predicted

    def evaluate_prediction(self, dates,x, y, evaluation_length, window_size, normalize, prediction_length):
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

            prediction = self.predict_sequence(data_initial, window_size,normalize,prediction_length)
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
        prediction_error = np.sqrt((prediction_mean_sq - np.power(prediction_mean,2)*(1-1/p_counter)))
        #print(prediction_mean,prediction_error)
        return(prediction_sign_mean,prediction_mean,prediction_error)


    '''***************************************DATA OPERATIONS*****************************************************'''

    def init_window_data(self,data,window_size,normalize):
        self.data_x = []
        self.data_y = []
        for d in data:
            for i in range(len(d)-window_size):
                x, y = self.get_next_window(d, i, window_size, normalize)
                self.data_x.append(x)
                self.data_y.append(y)

        print("[Model] Total data size is " + str(len(self.data_x)))
        return np.array(self.data_x), np.array(self.data_y)

    def inverse_transform_prediction(self, x_data, prediction_data):
        tranformed_predictions = []
        for i in range(len(prediction_data)):
            inputs = x_data[i]
            prediction = prediction_data[i][0]
            prediction_input = inputs[0]
            #print(prediction_input)
            #print(prediction)
            tranformed_predictions.append(prediction_input[0] * (1 + prediction))
        return (tranformed_predictions)

    def normalize_windows(self, window_data, single_window=False):
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

    def generate_sequential_batch(self,batch_size,shuffle):
        '''Yield a generator of training data from filename on given list of cols split for train/test'''
        data_length = np.shape(self.data_x)[0]
        window_size = np.shape(self.data_x)[1]
        indices = np.arange(data_length)
        #print("data x shape ",np.shape(self.data_x))
        if shuffle: np.random.shuffle(indices)
        i = 0
        while i < data_length:
            x_batch = []
            y_batch = []
            for b in range(batch_size):
                x = self.data_x[indices[i]]
                y = self.data_y[indices[i]]
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

    def generate_random_batch(self,batch_size):

        while True:
            batch_x = []
            batch_y = []
            indices = np.random.randint(0, np.shape(self.data_x)[0] - np.shape(self.data_x)[1], batch_size)
            #print(indices)
            for i in range(batch_size):
                # choose random index in features
                x = self.data_x[indices[i]]
                y = self.data_y[indices[i]]
                batch_x.append(x)
                batch_y.append(y)
            yield np.array(batch_x), np.array(batch_y)

    def get_next_window(self, data, i, seq_len, normalize):
        '''Generates the next data window from the given index location i'''
        window = data[i:i + seq_len]
        window = self.normalize_windows(window, single_window=True)[0] if normalize else window
        x = window[:-1]
        y = window[-1, [0]]
        return x, y

    '''***************************************SAVE / LOAD*****************************************************'''

    def save(self):
        self.model.save(self.abs_dir+"/models/rnn_" + self.name)
        print("[Model] " + self.name + " saved")

    def load(self):
        self.model = load_model(self.abs_dir+"/models/rnn_" + self.name)
        print("[Model] " + self.name + " loaded")