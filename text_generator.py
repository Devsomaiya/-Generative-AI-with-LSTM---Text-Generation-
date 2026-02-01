"""
LSTM Text Generator using Shakespeare Dataset
Author: Dev Somaiya
Description:
- Loads and preprocesses text data
- Builds and trains an LSTM model
- Generates new text based on seed input
"""

# =========================
# 1. IMPORT LIBRARIES
# =========================
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import string


# =========================
# 2. LOAD DATASET
# =========================
with open("shakespeare.txt", "r", encoding="utf-8") as file:
    text = file.read().lower()

print("Dataset loaded successfully!")


# =========================
# 3. PREPROCESSING
# =========================
# Remove punctuation
text = text.translate(str.maketrans("", "", string.punctuation))

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])

total_words = len(tokenizer.word_index) + 1
print("Total unique words:", total_words)

# Convert text to sequence of integers
token_list = tokenizer.texts_to_sequences([text])[0]

# =========================
# FIXED-LENGTH SLIDING WINDOW
# =========================
sequence_length = 40

input_sequences = []
for i in range(len(token_list) - sequence_length):
    input_sequences.append(token_list[i:i + sequence_length + 1])

input_sequences = np.array(input_sequences)

X = input_sequences[:, :-1]
y = input_sequences[:, -1]   # <-- NO ONE-HOT ENCODING

print("Preprocessing completed!")
print("Input shape:", X.shape)
print("Output shape:", y.shape)


# =========================
# 4. MODEL DESIGN
# =========================
model = Sequential([
    Embedding(
        input_dim=total_words,
        output_dim=100,
        input_length=sequence_length
    ),
    LSTM(100),
    Dense(total_words, activation='softmax')
])

model.compile(
    loss='sparse_categorical_crossentropy',  # 🔥 KEY FIX
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()


# =========================
# 5. MODEL TRAINING
# =========================
early_stop = EarlyStopping(
    monitor='loss',
    patience=3,
    restore_best_weights=True
)

model.fit(
    X,
    y,
    epochs=20,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1
)

print("Model training completed!")


# =========================
# 6. TEXT GENERATION FUNCTION
# =========================
def generate_text(seed_text, next_words=30):
    for _ in range(next_words):
        token_seq = tokenizer.texts_to_sequences([seed_text])[0]
        token_seq = pad_sequences(
            [token_seq],
            maxlen=sequence_length,
            padding='pre'
        )

        predicted_index = np.argmax(
            model.predict(token_seq, verbose=0),
            axis=-1
        )[0]

        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                seed_text += " " + word
                break

    return seed_text


# =========================
# 7. GENERATE SAMPLE OUTPUT
# =========================
print("\n--- Generated Text Samples ---\n")

seed_1 = "to be or not to be"
print("Seed:", seed_1)
print(generate_text(seed_1, 40))

print("\n-----------------------------\n")

seed_2 = "love is"
print("Seed:", seed_2)
print(generate_text(seed_2, 40))
