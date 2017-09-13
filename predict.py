from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

MAX_SEQUENCE_LENGTH = 1000

model = load_model('model.h5')

input = ["Very slim fitting could not get feet into shoes Very Very slim fitting"]

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

sequences = tokenizer.texts_to_sequences(input)
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
classes = model.predict(data)

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
print(classes)