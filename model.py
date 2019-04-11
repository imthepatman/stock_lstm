# Importing the Keras libraries and packages

import numpy as np
from numpy import newaxis
from keras.layers import Dense, Activation, Dropout, LSTM,GRU
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import os

class lstm:
    """A class for an building and inferencing an lstm model"""

    def build(self, configs):
        self.model = Sequential()

        for layer in configs['model']['layers']:
            neurons = layer['neurons'] if 'neurons' in layer else None
            dropout_rate = layer['rate'] if 'rate' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            return_seq = layer['return_seq'] if 'return_seq' in layer else None
            input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
            input_dim = layer['input_dim'] if 'input_dim' in layer else None

            if layer['type'] == 'dense':
                self.model.add(Dense(neurons, activation=activation))
            if layer['type'] == 'lstm':
                self.model.add(LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq))
            if layer['type'] == 'gru':
                self.model.add(GRU(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq))
            if layer['type'] == 'dropout':
                self.model.add(Dropout(rate = dropout_rate))

        self.model.compile(loss=configs['model']['loss'], optimizer=configs['model']['optimizer'])

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

    def train_generator(self, data,sequence_length,normalize, epochs, batch_size, steps_per_epoch):
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size, %s batches per epoch' % (epochs, batch_size, steps_per_epoch))

        self.model.fit_generator(
            self.generate_random_batch(data,sequence_length,batch_size,normalize),
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            workers=1
        )

        print('[Model] Training Completed')

    '''***************************************PREDICTION*****************************************************'''
    def predict_point_by_point(self, data, window_size, normalize):
        # Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
        x_data,y_data = self.get_data(data, window_size, False)
        x_norm, y_norm = self.get_data(data, window_size, True)
        print('[Model] Predicting Point-by-Point...')
        predicted = self.inverse_transform_prediction(x_data,self.model.predict(x_norm))
        #predicted = np.reshape(predicted, (predicted.size,))
        return predicted

    def predict_sequence(self, data_initial, window_size,normalize, prediction_len):
        # print("mulit predict for " + str(prediction_length) + " steps and window size "+ str(window_size))
        curr_frame = data_initial
        prediction = []
        for i in range(prediction_len):
            curr_frame_norm = self.normalize_windows(curr_frame, normalize)
            predicted = self.model.predict(curr_frame_norm)
            prediction.append(self.inverse_transform_prediction([curr_frame], predicted)[0])

            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size - 2], prediction[-1], axis=0)
        return prediction

    def predict_sequences_multiple(self, data,window_size, normalize, prediction_len):
        # Predict sequence of 50 steps before shifting prediction run forward by 50 steps
        x_data, y_data = self.get_data(data, window_size, False)
        print('[Model] Predicting Sequences Multiple...')
        prediction_seqs = []
        for i in range(int(len(data) / prediction_len)):
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

    def evaluate_prediction_sign(self,x,y,evaluation_length,window_size,normalize,prediction_length):
        reference_data = y[-evaluation_length-prediction_length:].flatten()
        right_prediction_counter = np.zeros(prediction_length)
        for pos in range(evaluation_length):
            if(pos % 10 == 0):
                print("evaluating prediction sign at position "+str(pos))
            data_initial = x[-pos-1]
            prediction = self.predict_sequence(data_initial, window_size,normalize,prediction_length)
            prediction = np.array(prediction).flatten()
            d_ref = np.array([reference_data[pos+i] - reference_data[pos-1] for i in range(prediction_length)])
            d_pred = np.array([prediction[i] - reference_data[pos-1] for i in range(prediction_length)])

            right_prediction_counter += np.where(np.multiply(d_ref,d_pred)>=0,1,0)
        return(right_prediction_counter/evaluation_length)


    '''***************************************DATA OPERATIONS*****************************************************'''

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

    def generate_random_batch(self, data, seq_len, batch_size, normalize):

            while True:
                batch_x = []
                batch_y = []
                indices = np.random.randint(0,len(data)-seq_len, batch_size)
                #print(indices)
                for i in range(batch_size):
                    # choose random index in features
                    x, y = self.get_next_window(data,indices[i], seq_len, normalize)
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

    def get_data(self, data,seq_len, normalize):
        '''
        Create x, y train data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise use generate_training_window() method.
        '''
        data_x = []
        data_y = []
        #+1 because i+1 value is target while i-seq_len:i is input
        for i in range(len(data)-seq_len):
            x, y = self.get_next_window(data, i, seq_len, normalize)
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)

    def test(self,x):
        print("test ",x)

    '''***************************************SAVE / LOAD*****************************************************'''

    def save(self, name):
        self.model.save("models/lstm_" + name)
        print("model " + name + " saved")

    def load(self, name):
        self.model = load_model("models/lstm_" + name)
        print("model " + name + " loaded")

'''***************************************PLOTTING*****************************************************'''

def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()


def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
	# Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()