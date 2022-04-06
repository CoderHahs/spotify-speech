# spotify-speech 🗣️

A speech recognizer for Spotify commands

## Project Summary

This project is intended to create a speech recognition model that can be used to recognize Spotify commands. In the future, this model will be connected to a computer where given voice input, the model will process and recognize the command and execute it.

## Methodology

### Training Data Creation

1. Recorded 10 sample commands
2. Used Spotify's Pedalboard Audio Processor, to change: Reverb, Pitch, Loudness
3. Created 2000 examples for each sample command

### Model Training

Used DeepSpeech 2-type architecture to train 2 models:

-   First model was training for 1 epoch
-   Second model was simpler, but trained for 5 epoch
-   Evaluated on dev set
-   Test the better model on test set

> Epoch count is low due to hardware restrictions and limited time, restricting available training time.

## Example of Creating Training Data with Pedalboard

Used Spotify's [Pedalboard](https://github.com/spotify/pedalboard) for audio processing.

```python
audio_file = AudioFile("data/play_spotify1.mp3", 'r')
audio = audio_file.read(audio_file.frames)
samplerate = audio_file.samplerate

board = Pedalboard([PitchShift(semitones=15)])

effected = board(audio, samplerate)

new_audio_file = AudioFile('data/processed-play_spotify1.wav', 'w', samplerate, effected.shape[0])
new_audio_file.write(effected)
```

## Model Structure - DeepSpeech 2

![DeepSpeech 2](https://nvidia.github.io/OpenSeq2Seq/html/_images/ds2.png)

### Simplified Model 1

```
______________________________________________________________________________________________________________
 Layer (type)                                    Output Shape                                Param #
==============================================================================================================
 input (InputLayer)                              [(None, None, 193)]                         0

 expand_dim (Reshape)                            (None, None, 193, 1)                        0

 conv_1 (Conv2D)                                 (None, None, 97, 32)                        14432

 conv_2 (Conv2D)                                 (None, None, 49, 32)                        236544

 conv_2_bn (BatchNormalization)                  (None, None, 49, 32)                        128

 conv_2_relu (ReLU)                              (None, None, 49, 32)                        0

 reshape (Reshape)                               (None, None, 1568)                          0

 dense_2 (Dense)                                 (None, None, 32)                            50208

==============================================================================================================
Total params: 301,312
Trainable params: 301,248
Non-trainable params: 64
______________________________________________________________________________________________________________
```

### Simplified Model 2

```
______________________________________________________________________________________________________________
 Layer (type)                                    Output Shape                                Param #
==============================================================================================================
 input (InputLayer)                              [(None, None, 193)]                         0

 expand_dim (Reshape)                            (None, None, 193, 1)                        0

 conv_1 (Conv2D)                                 (None, None, 97, 32)                        14432

 conv_2_bn (BatchNormalization)                  (None, None, 97, 32)                        128

 conv_2_relu (ReLU)                              (None, None, 97, 32)                        0

 reshape (Reshape)                               (None, None, 3104)                          0

 dense (Dense)                                   (None, None, 32)                            99360

==============================================================================================================
Total params: 113,920
Trainable params: 113,856
Non-trainable params: 64
______________________________________________________________________________________________________________
```

## Results

Model 2 performed better due to higher epoch count.

### Model 2 - Epoch

```
----------------------------------------------------------------------------------------------------
Word Error Rate: 0.8493
----------------------------------------------------------------------------------------------------
Target    : play spotify
Prediction: plauspouououogouotify
----------------------------------------------------------------------------------------------------
Target    : next song
Prediction: next snugugugugugugugugugugugugugouoguougugugugugus
----------------------------------------------------------------------------------------------------
Target    : skip song
Prediction: kip sngugugugugugugugugugugugugugugugugugugugugugugugu
----------------------------------------------------------------------------------------------------
Target    : reduce volume
Prediction: red ce vugugugugugugugugugugugugugugugugugugugugugugugu
----------------------------------------------------------------------------------------------------
Target    : pause music
Prediction: tauseougugugusic
----------------------------------------------------------------------------------------------------
```

## Next steps

Things to improve:

1. Train model 2 for longer (after getting hardware upgrade 😢, or use NVIDIA's cloud computing)
2. Implement execution on a computer

## References

-   [DeepSpeech2 — OpenSeq2Seq 0.2 documentation (nvidia.github.io)](https://nvidia.github.io/OpenSeq2Seq/html/speech-recognition/deepspeech2.html)
-   [Automatic Speech Recognition using CTC (keras.io)](https://keras.io/examples/audio/ctc_asr/)
-   [spotify/pedalboard: 🎛 🔊 A Python library for manipulating audio. (github.com)](https://github.com/spotify/pedalboard)

