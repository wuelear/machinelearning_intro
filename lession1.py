#http://machinelearningmastery.com/tutorial-first-neural-network-python-keras/

from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
import numpy


def create1():
    # create model
    model = Sequential()
    '''Creating a network with 8 inputs. The first layer has 12 neurons which are initialized via random
    method uniform. The activation function is a recitiver (relu)'''
    model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
    model.add(Dense(8, init='uniform', activation='relu'))
    model.add(Dense(1, init='uniform', activation='sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Fit the model
    model.fit(X, Y, nb_epoch=150, batch_size=10)

    model.save('model');

def create2():
    # create model
    model = Sequential()
    '''Creating a network with 8 inputs. The first layer has 12 neurons which are initialized via random
    method uniform. The activation function is a recitiver (relu)'''
    model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
    model.add(Dense(8, init='uniform', activation='relu'))
    model.add(Dense(1, init='uniform', activation='sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Fit the model
    model.fit(X, Y, nb_epoch=300, batch_size=10)

    model.save('model2');

def create3():
    # create model
    model = Sequential()
    '''Creating a network with 8 inputs. The first layer has 12 neurons which are initialized via random
    method uniform. The activation function is a recitiver (relu)'''
    model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
    model.add(Dense(8, init='uniform', activation='relu'))
    model.add(Dense(1, init='uniform', activation='sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Fit the model
    model.fit(X, Y, nb_epoch=150, batch_size=5)

    model.save('model3');


def create4():
    # create model
    model = Sequential()
    '''Creating a network with 8 inputs. The first layer has 12 neurons which are initialized via random
    method uniform. The activation function is a recitiver (relu)'''
    model.add(Dense(24, input_dim=8, init='uniform', activation='sigmoid'))
    model.add(Dense(16, init='uniform', activation='sigmoid'))
    model.add(Dense(8, init='uniform', activation='sigmoid'))
    model.add(Dense(1, init='uniform', activation='sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Fit the model
    model.fit(X, Y, nb_epoch=150, batch_size=10)

    model.save('model4')



# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load pima indians dataset
dataset = numpy.loadtxt("test.data", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

create4()

# model = load_model('model4')
#
#
# # evaluate the model
# scores = model.evaluate(X, Y)
# print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

