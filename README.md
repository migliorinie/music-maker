# Music Maker
A simple autoencoder and RNN music generator.
### Notes
train_convolver is the only file I've actually worked on.
###### How to launch train_convolver
`THEANO_FLAGS='floatX=float32,device=gpu1,lib.cnmem=0.9' python train_convolver.py indir netdir`
###### The sound quality is horrible!
Yes. It will improve later, after we are sure that the system works.

## What we need to do
1. Get the output from the autoencoder lambda and make it into one big array
2. Feed the big array to a LSTM network or other RNN
3. Restore the big array to wav files
4. Enjoy our new music

### Needed improvements
* Tina knows the best way to sample for the autoencoder.

### Optional things
* Using a library so that we can take mp3s and other files as in/out
