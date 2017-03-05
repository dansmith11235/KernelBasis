import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.python.framework import dtypes
from tensorflow.contrib import layers as tflayers
import dateutil.parser
import datetime
import matplotlib.dates as mdates
from tensorflow.contrib import learn
from sklearn.metrics import mean_squared_error

#reformat the data to be used in Rnn
#time steps is how many lags are used as input

def rnn_data(data, time_steps, labels=False):
    rnn_df = []
    for i in range(len(data) - time_steps):
        if labels:
            try:
                rnn_df.append(data.iloc[i + time_steps].as_matrix())
            except AttributeError:
                rnn_df.append(data.iloc[i + time_steps])
        else:
            data_ = data.iloc[i: i + time_steps].as_matrix()
            rnn_df.append(data_ if len(data_.shape) > 1 else [[i] for i in data_])

    return np.array(rnn_df, dtype=np.float32)

def rnn_data_test(data, time_steps, labels=False,ahead = 1):

    rnn_df = []
    for i in range(len(data) - time_steps - ahead - 1):
        if labels:
            try:
                rnn_df.append(data.iloc[i + time_steps + ahead - 1].as_matrix())
            except AttributeError:
                rnn_df.append(data.iloc[i + time_steps + ahead - 1])
        else:

            data_ = data.iloc[i: i + time_steps].as_matrix()
            rnn_df.append(data_ if len(data_.shape) > 1 else [[i] for i in data_])

    return np.array(rnn_df, dtype=np.float32)

def split_data(data, val_size=0.1, test_size=0.1):

    ntest = int(round(len(data) * (1 - test_size)))
    nval = int(round(len(data.iloc[:ntest]) * (1 - val_size)))

    df_train, df_val, df_test = data.iloc[:nval], data.iloc[nval:ntest], data.iloc[ntest:]

    return df_train, df_val, df_test


def prepare_data(data, time_steps, labels=False, val_size=0.1, test_size=0.1, ahead = 1):

    df_train, df_val, df_test = split_data(data, val_size, test_size)

    return (rnn_data(df_train, time_steps, labels=labels),
            rnn_data(df_val, time_steps, labels=labels),
            rnn_data_test(df_test, time_steps, labels=labels,ahead= ahead))

def load_csvdata(rawdata, time_steps, seperate=False, Ahead = 1):
    data = rawdata
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    train_x, val_x, test_x = prepare_data(data['a'] if seperate else data, time_steps, ahead = Ahead)
    train_y, val_y, test_y = prepare_data(data['b'] if seperate else data, time_steps, labels=True, ahead = Ahead)
    return dict(train=train_x, val=val_x, test=test_x), dict(train=train_y, val=val_y, test=test_y)

def load_frame(filename,col,datecol):
    #load the weather data and make a date
    data_raw = pd.read_csv(filename, dtype={datecol: str})
    data_raw[col] = data_raw[col].astype(float)
    data_raw[datecol] = pd.Series(data_raw[datecol], index=data_raw.index)
    df =  pd.DataFrame(data_raw, columns=[datecol,col])
    return df.set_index(datecol)

def lstm_cells(layers):

        return [tf.nn.rnn_cell.BasicLSTMCell(steps, state_is_tuple=True) for steps in layers]

def lstm_model(x):

    layer = {'weights':tf.Variable(tf.random_normal([RNN_LAYERS[-1], 1])),
                      'biases':tf.Variable(tf.random_normal([1]))}

    stacked_lstm = tf.nn.rnn_cell.MultiRNNCell(lstm_cells(RNN_LAYERS), state_is_tuple=True)

    x_ = tf.split(0, 1, x)

    outputs, states = tf.nn.rnn(stacked_lstm, x_, dtype=dtypes.float32)
        
    output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']

    return output

def train_LSTM(xTrain, yTrain,xTest, yTest, rnn_layers, Ahead = 1):
    
    prediction = lstm_model(X)

    cost = tf.reduce_mean(tf.square(prediction - Y))
    
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(epochs):
            epoch_loss = 0


            for _ in range(batch_size):

                RNum = np.random.randint(xTrain.shape[0])


                epoch_x = np.transpose(xTrain[RNum])

                epoch_y = yTrain[RNum]

                _, c = sess.run([optimizer, cost], feed_dict={X: epoch_x, Y: epoch_y})

                epoch_loss += c

            print('Epoch', epoch, 'completed out of',epochs,'loss:',epoch_loss)

        count = 1
        while count < Ahead:

            pred = prediction

            temp = pred.eval({X:xTest[:,:,0], Y:yTest})

            #print(temp)

            xTest = np.delete(xTest,0,1)

            xTest = np.insert(xTest,xTest.shape[1],temp,1)

            count = count + 1


        accuracy = tf.reduce_mean(tf.square(prediction -  Y))

        print('MSE:',accuracy.eval({X:xTest[:,:,0], Y:yTest}))

        pred = prediction

        np.savetxt("pred.csv", pred.eval({X:xTest[:,:,0], Y:yTest}), delimiter=",")

        np.savetxt("actual.csv", yTest, delimiter=",")


steps_ahead = 1

TIMESTEPS = 30

epochs = 100

batch_size = 100

RNN_LAYERS = [10, 10, 10]

StockData = load_frame("GE.csv",'RETX','date')

X = tf.placeholder('float', [None, TIMESTEPS])
Y = tf.placeholder('float')

Xinput, Yinput = load_csvdata(StockData, TIMESTEPS, seperate=False, Ahead = steps_ahead )

train_LSTM(Xinput['train'],Yinput['train'], Xinput['test'],Yinput['test'], RNN_LAYERS, Ahead = steps_ahead)
