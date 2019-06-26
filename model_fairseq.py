# Importing the Keras libraries and packages

import numpy as np
import tensorflow as tf
import os
import time

def encoder_block(inp, n_hidden, filter_size):
    inp = tf.expand_dims(inp, 2)
    inp = tf.pad(inp, [[0, 0], [(filter_size[0] - 1) // 2, (filter_size[0] - 1) // 2], [0, 0], [0, 0]])
    conv = tf.layers.conv2d(inp, n_hidden, filter_size, padding="VALID", activation=None)
    conv = tf.squeeze(conv, 2)
    return conv


def decoder_block(inp, n_hidden, filter_size):
    inp = tf.expand_dims(inp, 2)
    inp = tf.pad(inp, [[0, 0], [filter_size[0] - 1, 0], [0, 0], [0, 0]])
    conv = tf.layers.conv2d(inp, n_hidden, filter_size, padding="VALID", activation=None)
    conv = tf.squeeze(conv, 2)
    return conv


def glu(x):
    return tf.multiply(x[:, :, :tf.shape(x)[2] // 2], tf.sigmoid(x[:, :, tf.shape(x)[2] // 2:]))


def layer(inp, conv_block, kernel_width, n_hidden, residual=None):
    z = conv_block(inp, n_hidden, (kernel_width, 1))
    return glu(z) + (residual if residual is not None else 0)


class Model:
    def __init__(self,name):
        self.name = name
        self.abs_dir = os.path.dirname(os.path.realpath(__file__))


    def build(self, configs,load=False):
        self.config = configs
        self.X = tf.placeholder(tf.float32, (None, None,self.config['model']['input_size']))
        self.Y = tf.placeholder(tf.float32, (None, self.config['model']['output_size']))

        encoder_embedded = tf.layers.dense(self.X, self.config['model']['emb_size'])
        encoder_embedded = tf.nn.dropout(encoder_embedded, keep_prob=0.75)

        e = tf.identity(encoder_embedded)
        for i in range(self.config['model']['n_layers']):
            z = layer(encoder_embedded, encoder_block, 3, self.config['model']['n_hidden'] * 2, encoder_embedded)
            encoder_embedded = z

        encoder_output, output_memory = z, z + e
        g = tf.identity(encoder_embedded)

        for i in range(self.config['model']['n_layers']):
            attn_res = h = layer(encoder_embedded, decoder_block, 3, self.config['model']['n_hidden'] * 2,
                                 residual=tf.zeros_like(encoder_embedded))
            C = []
            for j in range(self.config['model']['n_attn_heads']):
                h_ = tf.layers.dense(h, self.config['model']['n_hidden'] // self.config['model']['n_attn_heads'])
                g_ = tf.layers.dense(g, self.config['model']['n_hidden'] // self.config['model']['n_attn_heads'])
                zu_ = tf.layers.dense(encoder_output, self.config['model']['n_hidden'] // self.config['model']['n_attn_heads'])
                ze_ = tf.layers.dense(output_memory, self.config['model']['n_hidden'] // self.config['model']['n_attn_heads'])

                d = tf.layers.dense(h_, self.config['model']['n_hidden'] // self.config['model']['n_attn_heads']) + g_
                dz = tf.matmul(d, tf.transpose(zu_, [0, 2, 1]))
                a = tf.nn.softmax(dz)
                c_ = tf.matmul(a, ze_)
                C.append(c_)

            c = tf.concat(C, 2)
            h = tf.layers.dense(attn_res + c, self.config['model']['n_hidden'])
            encoder_embedded = h

        encoder_embedded = tf.sigmoid(h)
        self.logits = tf.layers.dense(encoder_embedded[-1], self.config['model']['output_size'])
        self.cost = tf.reduce_mean(tf.square(self.Y - self.logits))
        self.optimizer = tf.train.AdamOptimizer(self.config['model']['learning_rate']).minimize(
            self.cost
        )

        self.sess = tf.InteractiveSession()
        if load:
            self.load()
        else:
            self.sess.run(tf.global_variables_initializer())

    def train_generator(self, x_train, y_train, epochs, batch_size, steps_per_epoch, shuffle=True, early_stopping_patience=1000, x_val=None, y_val=None):

        training_batch_generator = BatchGenerator(x_train, y_train, batch_size, shuffle)

        x_init,y_init = training_batch_generator.getitem(0)
        out_logits,out_cost = self.sess.run([self.logits,self.cost],
                                   feed_dict={
                                       self.X: x_init,
                                       self.Y: y_init
                                   },
                                   )
        print(np.shape(x_init),out_cost)

        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size, %s batches per epoch' % (epochs, batch_size, steps_per_epoch))
        time_start = time.time()

        print_skip = 10
        for i in range(epochs):
            total_loss = 0
            for k in range(0, training_batch_generator.len()):
                batch_x, batch_y = training_batch_generator.getitem(k)
                #shape_x = np.shape(batch_x)
                #batch_x = np.reshape(batch_x,(shape_x[0],shape_x[1]*shape_x[2]))
                _, loss = self.sess.run(
                    [self.optimizer, self.cost],
                    feed_dict={
                        self.X: batch_x,
                        self.Y: batch_y
                    },
                )
                total_loss += loss
                print("[Model] Epoch: " +str(i + 1)+" Trainingsample: "+str(k) +"/"+str(training_batch_generator.len()) + " loss: "+'{0:.6f}'.format(total_loss/(k+1))+"\t", end="\r")
            if (i + 1) % print_skip == 0:
                print('[Model] Epoch:', i + 1, 'Average loss: '+'{0:.6f}'.format(total_loss/training_batch_generator.len()) )
            training_batch_generator.on_epoch_end()

        print('[Model] Training completed in ' + '{0:.1f}'.format(time.time()-time_start) + "s")


    '''***************************************PREDICTION*****************************************************'''
    def predict(self,x_init):
        out_logits = self.sess.run(self.logits,
                              feed_dict={
                                  self.X: x_init
                              },
                              )
        return out_logits

    def predict_point_by_point(self, data, window_size, normalize,n_outputs):
                # Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
        x_data,_,_ = self.window_data(data, window_size, False,n_outputs)
        x_norm,_,_  = self.window_data(data, window_size, True,n_outputs)
        print('[Model] Predicting Point-by-Point...')
        predicted = self.inverse_transform_prediction(x_data,[self.predict(x_norm)[0,0]])
        #predicted = np.reshape(predicted, (predicted.size,))
        return predicted

    def predict_sequence(self, data_initial, window_size,normalize, prediction_len, n_outputs):
        # print("mulit predict for " + str(prediction_length) + " steps and window size "+ str(window_size))
        curr_frame = data_initial
        prediction = []
        for i in range(prediction_len):
            #print("frame ",i,curr_frame)
            curr_frame_norm = self.relative_normalize_window(curr_frame, normalize)
            predicted = self.predict(curr_frame_norm)[0,0]
            prediction.append(self.inverse_transform_prediction([curr_frame], [predicted])[0])

            curr_tmp = np.array(curr_frame[-1])
            curr_tmp[0] = prediction[-1]
            curr_frame = curr_frame[1:]
            #print(curr_tmp)
            curr_frame = np.insert(curr_frame, window_size - n_outputs-1, curr_tmp, axis=0)
        return prediction

    def predict_sequences_multiple(self, data,window_size, normalize, prediction_len, n_outputs):
        # Predict sequence of 50 steps before shifting prediction run forward by 50 steps
        x_data,_,_  = self.window_data(data, window_size, False,n_outputs)
        print('[Model] Predicting Sequences Multiple...')
        prediction_seqs = []
        for i in range(int(np.shape(data)[1] / prediction_len)-1):
            curr_frame = np.array(x_data[i * prediction_len])

            prediction = []
            for j in range(prediction_len):
                #print(curr_frame)
                curr_frame_norm = self.relative_normalize_window(curr_frame, normalize)
                #print(curr_frame_norm)
                predicted = self.predict(curr_frame_norm)[0,0]
                prediction.append(self.inverse_transform_prediction([curr_frame],[predicted])[0])
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
        saver = tf.train.Saver()
        saver.save(self.sess, self.abs_dir+"/models/tf_" + self.name)
        print("[Model] " + self.name + " saved")

    def load(self):
        saver = tf.train.Saver()
        saver.restore(self.sess, self.abs_dir + "/models/tf_" + self.name)
        print("[Model] " + self.name + " loaded")


class BatchGenerator():
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

    def len(self):
        'Denotes the number of batches per epoch'
        num_batches = int(np.floor(self.data_len / self.batch_size))
        return num_batches

    def getitem(self, index):
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