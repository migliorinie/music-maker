#Step 3: ottenere il corpus dal livello intermedio

import numpy as np
import os
import sys

import scipy.io.wavfile as wav
#import scipy.fftpack as fft
import keras
from keras.models import Sequential
from keras.layers.advanced_activations import PReLU
from keras.layers import Dense, Activation, Convolution1D, AveragePooling1D, UpSampling1D, Flatten, Reshape, LSTM, Lambda
from keras import backend as K

def roundlay(x):
  x = K.round(x)
  return x

# turning files into arrays should be done by a function
soundwavelist = [0 for i in range(20)]
direc = os.path.dirname(os.path.realpath(sys.argv[0]))
rate, soundwavelist[0] = wav.read(direc + "/samples_TH08/01.wav")
rate, soundwavelist[1] = wav.read(direc + "/samples_TH08/02.wav")
rate, soundwavelist[2] = wav.read(direc + "/samples_TH08/03.wav")
rate, soundwavelist[3] = wav.read(direc + "/samples_TH08/04.wav")
rate, soundwavelist[4] = wav.read(direc + "/samples_TH08/05.wav")
rate, soundwavelist[5] = wav.read(direc + "/samples_TH08/06.wav")
rate, soundwavelist[6] = wav.read(direc + "/samples_TH08/07.wav")
rate, soundwavelist[7] = wav.read(direc + "/samples_TH08/08.wav")
rate, soundwavelist[8] = wav.read(direc + "/samples_TH08/09.wav")
rate, soundwavelist[9] = wav.read(direc + "/samples_TH08/10.wav")
rate, soundwavelist[10] = wav.read(direc + "/samples_TH08/11.wav")
rate, soundwavelist[11] = wav.read(direc + "/samples_TH08/12.wav")
rate, soundwavelist[12] = wav.read(direc + "/samples_TH08/13.wav")
rate, soundwavelist[13] = wav.read(direc + "/samples_TH08/14.wav")
rate, soundwavelist[14] = wav.read(direc + "/samples_TH08/15.wav")
rate, soundwavelist[15] = wav.read(direc + "/samples_TH08/16.wav")
rate, soundwavelist[16] = wav.read(direc + "/samples_TH08/17.wav")
rate, soundwavelist[17] = wav.read(direc + "/samples_TH08/18.wav")
rate, soundwavelist[18] = wav.read(direc + "/samples_TH08/19.wav")
rate, soundwavelist[19] = wav.read(direc + "/samples_TH08/20.wav")

rate = 44100

#soundwave = fft.rfft(soundwave)

totl = 0
for i in range(len(soundwavelist)):
  tmp = np.zeros(((soundwavelist[i].shape[0]/rate + 1)*rate, 2))
  tmp[:soundwavelist[i].shape[0]] = soundwavelist[i]
  soundwavelist[i] = tmp
  totl += tmp.shape[0]/rate

wavelist = np.zeros((totl, rate, 2))
ctr = 0
for i in range(len(soundwavelist)):
  for j in range(soundwavelist[i].shape[0]/rate):
    wavelist[ctr] = soundwavelist[i][j*rate:(j+1)*rate]
    ctr += 1

model = Sequential()

#Training on the whole file is illogical since it has different dimensions. Cut shapes instead.
model.add(Convolution1D(4, 100, border_mode='same', subsample_length=5, input_shape=(44100, 2)))
#(8820, 4)
model.add(Convolution1D(4, 20, border_mode='same', subsample_length=2))
#(4410, 4)
model.add(Lambda(roundlay))

#(4410, 4)
model.add(UpSampling1D(length=5))
#(22050, 4)
model.add(Convolution1D(4, 50, border_mode='same'))
#(22050, 4)
model.add(UpSampling1D(length=2))
#(44100, 4)
model.add(Convolution1D(2, 100, border_mode='same'))

print("Compiling")
model.compile(optimizer="adadelta", loss="binary_crossentropy")
print("Compiled")

#model.fit(wavelist, wavelist, batch_size=4, nb_epoch=8, verbose=1)
model.fit(wavelist, wavelist, batch_size=4, nb_epoch=1, verbose=1)

model.save(direc + "/models/convolver.h5")

rate, soundtest = wav.read(direc + "/samples_TH08/04.wav")

tmp = np.zeros(((soundtest.shape[0]/rate + 1)*rate, 2))
tmp[: soundtest.shape[0]] = soundtest
soundtest = tmp

soundlist = np.zeros((soundtest.shape[0]/rate, rate, 2))
for j in range(soundlist.shape[0]):
  soundlist[j] = soundtest[j*rate:(j+1)*rate]

output = model.predict(soundlist)

outfile = np.zeros((soundlist.shape[0]*rate, 2))
for i in range(len(output)):
    outfile[i*44100:(i+1)*44100] = output[i]

outfile = outfile/outfile.max()

wav.write('test1.wav', 44100, outfile)


get_3rd_layer_output = K.function([model.layers[0].input],
                                  [model.layers[2].output])
layer_output = get_3rd_layer_output([soundlist])[0]

print(layer_output.shape)

orig_shape=layer_output.shape

remade_output=layer_output.ravel()
remade_output=remade_output.reshape(orig_shape)
print(np.array_equal(remade_output, layer_output))
