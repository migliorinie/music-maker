# NDM: Autoencoder. Dal livello 3 si ottiene l'array compresso, fatto da interi.
#Step 3: ottenere il corpus dal livello intermedio

import numpy as np
import os
import sys
import argparse

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
def main(args):
    #First, I read the wav files and pus them into the soundwavelist array list
    for filename in os.listdir(args.indir):
        if not filename.endswith(".wav"):
            print("Error! The folder must only contain .wav files!")
            return
    soundwavelist = [0 for i in range(len(os.listdir(args.indir)))]
    i = 0
    for filename in os.listdir(args.indir):
        rate, soundwavelist[i] = wav.read(args.indir + '/' + filename)
        i += 1
    rate = 44100

    #soundwave = fft.rfft(soundwave)

    # Here I set the list of music files to a single, large array
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

    # Time to neural network
    model = Sequential()

    #Training on the whole file is illogical since it has different dimensions. Cut shapes instead.
    model.add(Convolution1D(4, 100, border_mode='same', subsample_length=5, input_shape=(44100, 2)))
    #(8820, 4)
    model.add(Convolution1D(4, 20, border_mode='same', subsample_length=2))
    #(4410, 4)
    # model.add(Lambda(roundlay, output_shape=(None, 4410, 4))) # This way, it crashes.
    model.add(Lambda(roundlay)) # This way, it throws a warning
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

    # Training the network and saving
    model.fit(wavelist, wavelist, batch_size=4, nb_epoch=1, verbose=1)

    model.save(args.netfile)

    # Here I make a test to check the quality.
    if args.testfile != 'notest':
        rate, soundtest = wav.read(args.testfile)
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

        wav.write(args.testout, 44100, outfile)


    get_3rd_layer_output = K.function([model.layers[0].input],
                                      [model.layers[2].output])
    layer_output = get_3rd_layer_output([soundlist])[0]

    print(layer_output.shape)

    orig_shape=layer_output.shape

    remade_output=layer_output.ravel()
    remade_output=remade_output.reshape(orig_shape)
    print(np.array_equal(remade_output, layer_output))

def get_parser():
    parser = argparse.ArgumentParser(description="""
    Trains a convolution autoencoder from a folder of .wav files
    """, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('indir', metavar='indir', type=str, help='The folder where the input files are found')
    parser.add_argument('netfile', metavar='netfile', type=str, help='The file where to save the autoencoder')
    parser.add_argument('--testfile', metavar='testfile', dest='testfile', action='store', type=str, default='notest', help='Input file to test processing')
    parser.add_argument('--testout', metavar='testout', dest='testout', action='store', type=str, default='test1.wav', help='Output file to save test')
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
