import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
import numpy as np
import pandas as pd


files = ['feb99', 'july99', 'feb00', 'july00', 'feb01', 'july01', 'feb02', 'july02',
        'feb03', 'july03', 'feb04', 'july04', 'feb05', 'july05', 'feb06', 'july06',
        'feb07', 'july07', 'feb08', 'july08', 'feb09', 'july09', 'feb10', 'july10',
        'feb11', 'july11', 'feb12', 'july12', 'feb13', 'july13', 'feb14', 'july14',
        'feb15', 'july15', 'feb16']

mydata = "FOMCSentimentAnalysis.xlsx"

maxlen = 20000
training_samples = 30
validation_samples = 5
max_words = 20000
embedding_dim = 150

samples = []
for x in files:
    sample = open(x, 'r',encoding='utf-8')
    sample = sample.read()
    samples.append(sample)

labels = []
excel_data = pd.ExcelFile(mydata)
excel_sheet = excel_data.parse('Sheet1')
eur_usd_change = excel_sheet["EUR/USD_Change"]
labels = np.asarray(eur_usd_change)


tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(samples)

sequences = tokenizer.texts_to_sequences(samples)

#one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=maxlen)

print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]

callbacks_list = [
    keras.callbacks.ModelCheckpoint(
    filepath='FOMC_Analysis_v1.h5',
    monitor='val_loss',
    save_best_only=True,
    )
]

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer='adam',
loss='binary_crossentropy',
metrics=['acc'])
model.fit(x_train, y_train,
epochs=10,
batch_size=32,
callbacks=callbacks_list,
validation_data=(x_val, y_val))

model.load_weights('FOMC_Analysis_v1.h5')
score_val = model.evaluate(x_val, y_val)
print('Test loss:', score_val[0])
print('Test accuracy:', score_val[1])

files_test = ['july16', 'feb17', 'july17','feb18']

mydata_test = "FOMCSentimentAnalysis_Test.xlsx"

samples_test = []
for x in files_test:
    sample_test = open(x, 'r',encoding='utf-8')
    sample_test = sample_test.read()
    samples_test.append(sample)

labels_test = []
excel_data_test = pd.ExcelFile(mydata_test)
excel_sheet_test = excel_data_test.parse('Sheet1')
eur_usd_change_test = excel_sheet_test["EUR/USD_Change"]
labels_test = np.asarray(eur_usd_change_test)

sequences_test = tokenizer.texts_to_sequences(samples_test)

x_test = pad_sequences(sequences_test, maxlen=maxlen)
y_test = np.asarray(labels_test)

model.load_weights('FOMC_Analysis_v1.h5')
score = model.evaluate(x_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
