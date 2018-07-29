# Artificial-Intelligence-FOMC-Monetary-Policy-Analysis-for-EUR-USD

This is the first project I've decided to do in Artificial Intelligence. I've started getting quite a passion for the subject and its potential applications. This uses natural language processing (NLP) for binary classification from a multi-layer perceptron (MLP). Since this is my first project, the model is quite elementary and doesn't provide anything statistically significant on the test set (It doesn't provide a higher level of accuracy than 50%). Never-the-less, by using more advanced features in Keras, I've already gotten a statistically significant result on my test set with another model I'm currently working on.


Using TensorFlow backend.
Found 8879 unique tokens.
Shape of data tensor: (35, 20000)
Shape of label tensor: (35,)
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_1 (Embedding)      (None, 20000, 150)        3000000   
_________________________________________________________________
flatten_1 (Flatten)          (None, 3000000)           0         
_________________________________________________________________
dense_1 (Dense)              (None, 32)                96000032  
_________________________________________________________________
dense_2 (Dense)              (None, 16)                528       
_________________________________________________________________
dense_3 (Dense)              (None, 4)                 68        
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 5         
=================================================================
Total params: 99,000,633
Trainable params: 99,000,633
Non-trainable params: 0
_________________________________________________________________
Train on 30 samples, validate on 5 samples
Epoch 1/10
2018-07-28 17:44:28.161931: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2
30/30 [==============================] - 4s 150ms/step - loss: 0.6951 - acc: 0.4000 - val_loss: 0.9779 - val_acc: 0.2000
Epoch 2/10
30/30 [==============================] - 3s 116ms/step - loss: 0.4317 - acc: 0.7667 - val_loss: 1.1679 - val_acc: 0.2000
Epoch 3/10
30/30 [==============================] - 3s 115ms/step - loss: 0.1396 - acc: 0.9333 - val_loss: 0.9146 - val_acc: 0.2000
Epoch 4/10
30/30 [==============================] - 3s 108ms/step - loss: 0.0227 - acc: 1.0000 - val_loss: 0.7182 - val_acc: 0.4000
Epoch 5/10
30/30 [==============================] - 3s 108ms/step - loss: 0.0084 - acc: 1.0000 - val_loss: 0.6026 - val_acc: 0.8000
Epoch 6/10
30/30 [==============================] - 3s 107ms/step - loss: 0.0033 - acc: 1.0000 - val_loss: 0.5490 - val_acc: 0.8000
Epoch 7/10
30/30 [==============================] - 3s 107ms/step - loss: 0.0014 - acc: 1.0000 - val_loss: 0.5170 - val_acc: 0.8000
Epoch 8/10
30/30 [==============================] - 3s 108ms/step - loss: 6.2672e-04 - acc: 1.0000 - val_loss: 0.4956 - val_acc: 0.8000
Epoch 9/10
30/30 [==============================] - 3s 106ms/step - loss: 2.9860e-04 - acc: 1.0000 - val_loss: 0.4828 - val_acc: 0.8000
Epoch 10/10
30/30 [==============================] - 3s 110ms/step - loss: 1.5072e-04 - acc: 1.0000 - val_loss: 0.4755 - val_acc: 0.8000
5/5 [==============================] - 0s 24ms/step
Test loss: 0.4754864275455475
Test accuracy: 0.800000011920929
4/4 [==============================] - 0s 23ms/step
Test loss: 7.971192359924316
Test accuracy: 0.5

