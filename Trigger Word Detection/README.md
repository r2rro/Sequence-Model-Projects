## Trigger Word Detection
The goal of this project is to:

* Structure a speech recognition project
* Synthesize and process audio recordings to create train/dev datasets
* Train a trigger word detection model and make predictions

## Data Synthesis

A speech dataset should ideally be as close as possible to the real application. In this case, we like to detect the word "activate" in working environments (library, home, offices, open-spaces ...).
Therefore, we need to create recordings with a mix of positive words ("activate") and negative words (random words other than activate) on different background sounds.

In the raw_data directory, you can find a subset of the raw audio files of the positive words, negative words, and background noise. These audio files  will be used to synthesize a dataset to train the model.
* The "activate" directory contains positive examples of people saying the word "activate".
* The "negatives" directory contains negative examples of people saying random words other than "activate".
* There is one word per audio recording.
* The "backgrounds" directory contains 10 second clips of background noise in different environments.

## Model Architecture
Our goal is to build a network that will ingest a spectrogram and output a signal when it detects the trigger word. This network will use 4 layers:

* A convolutional layer
* Two GRU layers
* A dense layer. 

Here is the architecture we will use:
<p align="center">
  <img width="500" src="https://github.com/r2rro/Sequence-Model-Projects/blob/main/Trigger%20Word%20Detection/images/model.png" alt="gru">
</p>
