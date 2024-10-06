# MNIST-GAN-using-PyTorch
This repository contains a PyTorch implementation of a Generative Adversarial Network (GAN) for generating handwritten digits from the MNIST dataset. The code is written in a Jupyter Notebook, and it walks through building, training, and visualizing results.

## Introduction

A Generative Adversarial Network (GAN) consists of two models:

	1.	Generator: Learns to generate new, plausible data.
	2.	Discriminator: Evaluates the authenticity of the generated data.

The two models are trained simultaneously in a game-like scenario where the generator improves at creating data, and the discriminator improves at identifying generated data from real data.

In this notebook, we build a GAN to generate images of handwritten digits that resemble those in the MNIST dataset.

## Requirements

Install the required libraries using the following command:
```
pip install torch torchvision matplotlib numpy
```

## Dataset

The MNIST dataset contains 70,000 images of handwritten digits (0-9), split into training and testing sets. Each image is 28x28 pixels, grayscale.

In this project, the dataset is loaded using torchvision.datasets.MNIST, which automatically downloads and processes the data.

## Model Architecture

	•	Generator: Takes random noise as input and generates a 28x28 pixel image.
	•	Discriminator: Takes an image as input and classifies it as either real or fake.

The generator and discriminator are built using fully connected layers with activation functions such as LeakyReLU.


## Training

The GAN is trained over several epochs with the following process:

	1.	The generator generates fake images.
	2.	The discriminator evaluates both real and fake images.
	3.	Losses from both models are computed, and gradients are backpropagated.
	4.	The process repeats, improving both models over time.

## Results

After training, the generator can create realistic images of handwritten digits that resemble the MNIST dataset. Sample results are displayed at the end of the notebook.


## Usage
 1. Clone the repository:
```
git clone https://github.com/yourusername/mnist-gan-pytorch.git
cd mnist-gan-pytorch

```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Open the jupyter notebook:
   ```
   jupyter notebook MNIST_GAN.ipynb
   ```
4.	Run all the cells to train the GAN and visualize the generated images.


 
