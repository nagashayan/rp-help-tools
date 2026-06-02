import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight


X = np.load("X_gru.npy")
y = np.load("y_gru.npy")

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Class counts:", np.bincount(y))

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

classes = np.unique(y_train)
weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train,
)
class_weight = dict(zip(classes, weights))
print("Class weights:", class_weight)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1], X.shape[2])),
    tf.keras.layers.Masking(mask_value=0.0),
    tf.keras.layers.GRU(32),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
    )
]

model.fit(
    X_train,
    y_train,
    validation_split=0.25,
    epochs=100,
    batch_size=8,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1,
)

pred_prob = model.predict(X_test).ravel()
pred = (pred_prob >= 0.5).astype(int)

print("\nConfusion Matrix")
print(confusion_matrix(y_test, pred))

print("\nClassification Report")
print(classification_report(y_test, pred, zero_division=0))

model.save("gru_handshake_sequence.keras")
print("Saved gru_handshake_sequence.keras")
