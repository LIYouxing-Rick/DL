from glob import glob
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
import jieba
import re
from tensorflow.keras.preprocessing import *
import gensim
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D, SpatialDropout1D, LSTM, Concatenate
from tensorflow.keras import Model
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report

# embedding for the text(in Chinese form),here youcan use eny pre-trained word2vec model
fd = 'tab the path where you restore the pre-training word_split '  
model = gensim.models.KeyedVectors.load_word2vec_format(fd, encoding="utf-8", limit=100000)  


# Take the cut_review column of the df and cut it by space to find the corresponding word vector for each word in the model
df['embeds'] = df['cut_review'].apply(lambda sentence: np.array([model[word] for word in sentence.split(" ") if word in model.index2word]))

# Because word2vec after the length of the vec varies, here the uniform use of 0 patch, because the maximum length does not exceed 20, here choose to patch into the length of 20
def padding(x, MAX_LENS=MAX_LENS):   
    y = np.zeros((MAX_LENS - x.shape[0], EMBEDDING_DIM)) 
    z = np.concatenate((x, y), axis=0)                  
    return z

titles = df['embeds'].apply(lambda x: padding(x))      
titles = np.array(list(titles))                         

print(np.shape(titles))
print(titles[0, 0].shape)  
print(titles[0, 0, 0:10])  




# The normalization for the price
prices = df[["price"]].values 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
prices_scaled = scaler.fit_transform(prices).astype(np.float32)
p=prices_scaled 
print(p.dtype)  

print(pics.shape, p,titles.shape, Y.shape)                                               # 这里是我们的自变量（图片、标题）和因变量（销量分类）



# The model setup for the image
from tensorflow.keras.applications.inception_v3 import InceptionV3
vgg_model = Xception(weights='imagenet', include_top=False, input_shape=(IMSIZE, IMSIZE, 3))

for layers in vgg_model.layers:
    layers.trainable = False            


# feature extraction for the image
output1 = Flatten()(vgg_model.output)    
model1 = Model(vgg_model.input, output1) 
model1.summary()                        

# feature extraction for the text
input2 = Input(shape=(MAX_LENS, EMBEDDING_DIM))       
x = SpatialDropout1D(0.2)(input2)                     
x = LSTM(100, dropout=0.2, recurrent_dropout=0.2)(x)  
x = Dense(32, activation='softmax')(x)                
output2 = x                                          
model2 = Model(input2, output2)                       
model2.summary()                                     


price_input = Input(shape=(1,), name="price_input") 

# fusion for the multi-modal feature 
x = tf.concat([output1, output2,price_input], axis=1)  
x = Dense(512, activation='relu')(x)      
x = Dense(3, activation='softmax')(x)     
# 最终模型
final_model = Model(inputs=[model1.inputs[0], model2.inputs[0],price_input],outputs=x)  



# the setting of training parameter

pics_train, pics_test, titles_train, titles_test,p_train,p_test, Y_train, Y_test = train_test_split(pics, titles,p, Y, test_size=0.2, random_state=0)

final_model.compile(loss='sparse_categorical_crossentropy',          
                    optimizer=Adam(0.001),                           
                    metrics=['accuracy'])                           
final_model.fit([pics_train, titles_train,p_train], Y_train,                
                validation_data=([pics_test, titles_test,p_test], Y_test),  
                batch_size=64, epochs=10,                            
                workers=12)                                          

y_test_predict = final_model.predict([pics_test, titles_test,p_test])  # Model Prediction Probability
y_test_predict = np.argmax(y_test_predict, axis=1)              # Take the one with the highest probability as the final predicted label
print("VGG16", classification_report(Y_test, y_test_predict))   # View Prediction Accuracy


# The visualization for the prediction 
fig, ax = plt.subplots(rows, cols)                
fig.tight_layout()                                
for i in range(rows):                             
    for j in range(cols):                            
        ax[i, j].imshow(pics_test[i * rows + j]) 
        ax[i, j].set_title(f"Label{Y_test[i * rows + j]}; Predict{y_test_predict[i * rows + j]}")  

