# DL
my dl project


# 1.taobao_E_commerce_prediction
The program is designed by Tensorflow frame work

## Multi-Modal Taobao Sales Prediction
This repository contains a multi-modal deep learning model for predicting product sales categories on Taobao (Chinese e-commerce platform) using image, text, and price data.

## Model Overview
The model combines three modalities:

**​1.​Image Features**​​: Extracted using pre-trained VGG16
​**​2.Text Features​​**: Processed with LSTM using pre-trained Chinese word embeddings
**​​3.Price Data**​​: Normalized numerical feature

Features are fused through concatenation and processed by dense layers for final classification.

## Features
- Multi-modal fusion (image + text + numerical)
- Pre-trained CNNs for image understanding
- LSTM for Chinese text processing
- Customizable embedding layer for Chinese vocabulary
- StandardScaler for price normalization

## Requirements
```bash
Python 3.7+
tensorflow==2.5.0
keras==2.5.0
gensim==4.0.1
jieba==0.42.1
pandas==1.3.0
numpy==1.19.5
matplotlib==3.4.3
scikit-learn==0.24.2
Pillow==8.3.1
