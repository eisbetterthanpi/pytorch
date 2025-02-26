
Learn Pytorch, from noob to ninja!

<!-- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eisbetterthanpi/python/pytorch/blob/master/Autoencoder.ipynb) -->

## Contents
1. [learn_fn](#learn_fn)
2. [base](#base)
3. [CNN](#cnn)
4. [RNN2](#rnn2)
5. [Autoencoder](#autoencoder)
6. [vae](#vae)
7. [GAN](#gan)
8. [gpt2_archive_uci](#gpt2_archive_uci)
<!-- 4. [Fourth Example](#fourth-examplehttpwwwfourthexamplecom) -->

## learn_fn
basic feed forward neural network<br />
learn a algebriac function<br />
nn as function approximators. backprop loss, optimizer step
#### [Open `base.ipynb` in Google Colab](https://colab.research.google.com/github/eisbetterthanpi/pytorch/blob/master/base.ipynb)


## base
basic feed forward neural network<br />
classifying FashionMNIST dataset<br />
flatten image before passing to model with nn.Sequential linear layers
#### [Open `base.ipynb` in Google Colab](https://colab.research.google.com/github/eisbetterthanpi/pytorch/blob/master/base.ipynb)

## CNN
basic convolutional neural network<br />
classifying CIFAR10 dataset<br />
model with nn.Conv2d and nn.MaxPool2d followed by a flattened Linear layer
pure convolution
#### [Open `CNN.ipynb` in Google Colab](https://colab.research.google.com/github/eisbetterthanpi/pytorch/blob/master/CNN.ipynb)


## RNN2
recurrent neural network

#### [Open `RNN2.ipynb` in Google Colab](https://colab.research.google.com/github/eisbetterthanpi/pytorch/blob/master/RNN2.ipynb)

<div align="center">
  <div>&nbsp;</div>
  <img src="resources/rnn.png" height="200"/>
  <div align="center">img: dprogrammer.org/rnn-lstm-gru</div>
</div>


## Autoencoder
learn to recreate mnist image<br />
learning rate schedulers(very important!)<br />
#### [Open `Autoencoder.ipynb` in Google Colab](https://colab.research.google.com/github/eisbetterthanpi/pytorch/blob/master/Autoencoder.ipynb)
<div align="center">
  <div>&nbsp;</div>
  <img src="resources/ae_og.png" width="200"/> 
  <img src="resources/ae_re.png" width="200"/> 
  <div align="center">original | reconstructed </div>
</div>


## Convolutional Autoencoder
learn to recreate mnist image using convolutional autoencoder<br />
model size is significantly smaller than the naive MLP autoencoder<br />
#### [Open `conv_autoencoder.ipynb` in Google Colab](https://colab.research.google.com/github/eisbetterthanpi/pytorch/blob/master/conv_autoencoder.ipynb)
<div align="center">
  <div>&nbsp;</div>
  <img src="resources/og4.png" width="200"/> 
  <img src="resources/convt4.png" width="200"/> 
  <img src="resources/upsample4.png" width="200"/> 
  <div align="center">original | conv transpose | conv upsample </div>
</div>
two ways of doing deconvolution: nn.ConvTranspose2d and nn.Upsample

## vae
variational autoencoder
#### [Open `vae.ipynb` in Google Colab](https://colab.research.google.com/github/eisbetterthanpi/pytorch/blob/master/vae.ipynb)
<div align="center">
  <div>&nbsp;</div>
  <img src="resources/vae.png" width="400"/>
  <div align="center">original | reconstructed </div>
</div>

<!-- 
[Paper](https://arxiv.org/abs/2312.01479) |
[Website](https://research.myshell.ai/open-voice) 
[Video](https://github.com/myshell-ai/OpenVoice/assets/40556743/3cba936f-82bf-476c-9e52-09f0f417bb2f)
 -->

## GAN
Generative Adversarial Network (GAN)
training and generating emojis
#### [Open `dcgan.ipynb` in Google Colab](https://colab.research.google.com/github/eisbetterthanpi/pytorch/blob/master/dcgan.ipynb)

<div align="center">
  <div>&nbsp;</div>
  <img src="resources/gan_arch.png" height="200"/> 
  <img src="resources/gan.jpg" height="200"/> 
  <div align="center">img: sthalles.github.io/intro-to-gans</div>
</div>

training on emoji dataset to generate 

## gpt2_archive_uci
fine tuning pretrained huggingface gpt2 for text classification
#### [Open `gpt2_archive_uci.ipynb` in Google Colab](https://colab.research.google.com/github/eisbetterthanpi/pytorch/blob/master/gpt2_archive_uci.ipynb)

<div align="center">
  <div>&nbsp;</div>
  <img src="resources/gpt_train.png" height="200"/> 
  <img src="resources/gpt_matrix.png" height="200"/> 
  <div align="center">img: sthalles.github.io/intro-to-gans</div>
</div>

<!--  -->

If you find this code useful, please credit `eisbetterthanpi`
<!-- [website](https://github.com/eisbetterthanpi) -->
[website]: https://github.com/eisbetterthanpi

