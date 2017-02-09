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
direc = os.path.dirname(os.path.realpath(sys.argv[0]))

rate, dirty = wav.read(direc + "/test1.wav")
rate, clean = wav.read(direc + "/imperishable/04.wav")

#No increase because I know it's precise, since it was fed by the other network.
tmp = np.zeros(((dirty.shape[0]/rate)*rate, 2))
tmp[: dirty.shape[0]] = dirty
dirty = tmp

dirty_input = np.zeros((dirty.shape[0]/rate, rate, 2))
for j in range(dirty_input.shape[0]):
  dirty_input[j] = dirty[j*rate:(j+1)*rate]

tmp = np.zeros(((clean.shape[0]/rate + 1)*rate, 2))
tmp[: clean.shape[0]] = clean
clean = tmp

clean_output = np.zeros((clean.shape[0]/rate, rate, 2))
for j in range(clean_output.shape[0]):
  clean_output[j] = clean[j*rate:(j+1)*rate]

rate = 44100

#soundwave = fft.rfft(soundwave)

model = Sequential()

#Training on the whole file is illogical since it has different dimensions. Cut shapes instead.
model.add(Convolution1D(2, 100, border_mode='same', input_shape=(44100, 2)))

model.add(Convolution1D(2, 50, border_mode='same'))

model.add(Convolution1D(2, 20, border_mode='same'))

print("Compiling")
model.compile(optimizer="adadelta", loss="binary_crossentropy")
print("Compiled")

model.fit(dirty_input, clean_output, batch_size=4, nb_epoch=8, verbose=1)

# I should probably split everything and save the model, so that it won't go to heck

#Need to make a GT through the convolver.
#rate, soundtest = wav.read(direc + "/imperishable/04.wav")
rate, soundtest = wav.read(direc + "/test1.wav")

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

wav.write(direc + '/reconTest.wav', 44100, outfile)

