#NdM: Questo era un tentativo di adattare il generatore di testo. Ma non mi ricordo come funzioni.

'''Example script to generate text from Nietzsche's writings.
At least 20 epochs are required before the generated text
starts sounding coherent.
It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.
If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''

from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys

import scipy.io.wavfile as wav
import scipy.fftpack as fft

rate, soundwave = wav.read('samples_TH08/09.wav')
cha1 = np.zeros((soundwave.shape[0]//rate+1)*rate)
cha1[:soundwave.shape[0]] = soundwave[:, 0]
fourier = fft.rfft(cha1)

excon = False
m = max(fourier)
curr = fourier.shape[0]-1
while excon == False:
  if fourier[curr] >= m*0.1:
    excon = True
  else:
    curr=curr-1

for i in range(curr, fourier.shape[0]-1):
  fourier[i] = 0
print(str(fourier.shape[0]-curr) + " frequencies out of " + str(fourier.shape[0]) + " cut")

text = fourier[:curr]

print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# build the model: 2 stacked LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(512, return_sequences=True, input_shape=(maxlen, len(chars))))
model.add(LSTM(512, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')


def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

# train the model, output generated text after each iteration
for iteration in range(1, 60):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, y, batch_size=128, nb_epoch=1)

    start_index = random.randint(0, len(text) - maxlen - 1)

    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('----- diversity:', diversity)

        sentence = text[start_index: start_index + maxlen]

        outfile = np.zeros(44100+maxlen)
        outfile[:maxlen] = sentence

        for i in range(44100):
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            outfile[i+maxlen] = next_char
            sentence = sentence[1:] + next_char

        filename = "out_" + str(i) + "_" + str(diversity) + ".wav"
        outfile = fft.irfft(outfile)
        wav.write(filename, 44100, outfile)

        print("File " + filename + " saved")
