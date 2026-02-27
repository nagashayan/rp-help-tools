from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, GRU
import numpy as np
import os

# 1. Load Data
DATA_PATH = os.path.join('MP_Data') 
actions = np.array(['handshake', 'background'])
label_map = {label:num for num, label in enumerate(actions)}
sequences, labels = [], []

for action in actions:
    for sequence in range(30): # no_sequences
        window = []
        for frame_num in range(30): # sequence_length
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

# 2. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# 3. Build the Lightweight GRU Model
model = Sequential()

# Layer 1: GRU (Gated Recurrent Unit) - Faster than LSTM
# return_sequences=True because we stack another layer
model.add(GRU(64, return_sequences=True, activation='relu', input_shape=(30,63))) 

# Layer 2: Second GRU layer
# return_sequences=False because next is a Dense layer
model.add(GRU(32, return_sequences=False, activation='relu'))

# Layer 3: Dense layers for decision
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))

# Output Layer: Softmax for probability
model.add(Dense(actions.shape[0], activation='softmax'))

# 4. Compile and Train
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Train for 50 epochs (adjust based on your data size)
model.fit(X_train, y_train, epochs=50, callbacks=[])

# 5. Save the weights
model.save('handshake_gru_v1.h5')
print("Model Saved!")
