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



#  2.Classical Chinese Poetry Generation with RNN

This repository contains a recurrent neural network (RNN) model for generating and completing classical Chinese poetry. The model learns patterns from Tang poetry corpus and can generate new poems or complete partial verses in traditional Chinese literary style.

## Model Architecture

### Key components:
• Embedding Layer: 640-dimensional word embeddings

• RNN Layer: 1280-unit SimpleRNN with sequence output

• Dense Layer: Vocabulary-sized output with softmax activation


### Features
• Traditional Chinese character-level text generation

• Automatic verse completion with placeholder handling

• Sequence padding and truncation for fixed-length processing

• Customizable poetry length (default: 50 characters)

• Word index mapping with Keras Tokenizer


## Requirements
```bash
Python 3.7+
tensorflow==2.5.0
numpy==1.19.5
scikit-learn==0.24.2



## Dataset Preparation
1. Poetry text file format:
```
Title:PoemContent
Title:PoemContent
...
```
2. Example line:
`春晓:春眠不觉晓 处处闻啼鸟 夜来风雨声 花落知多少`
3. Preprocessing automatically handles:
• Title/content separation

• Character tokenization

• Sequence padding (post-padding to 50 characters)


## Performance
Typical training metrics:
```
Epoch 20/20
loss: 1.2324 - accuracy: 0.6821
val_loss: 2.1145 - val_accuracy: 0.5523
```

Generation Examples
Input Template | Generated Verse
--------------|-----------------
`春*不觉晓` | `春眠不觉晓`
`床前**光` | `床前明月光`
`海内***涯` | `海内存知己 天涯若比邻`

License
MIT License

Notes
• The asterisk `*` acts as placeholder for character prediction

• Default temperature setting uses argmax for conservative generation

• For creative variations, modify prediction sampling strategy



Important: Replace placeholder diagram and update file paths in the code. Adjust `embedding_size` and `hidden_size` parameters according to your computational resources and dataset size.
