import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

# ============================================================
#         EMOTION BASED MOVIE RECOMMENDATION - TRAINING
# ============================================================

# Image size for MobileNetV2
IMG_SIZE = (96, 96)
BATCH_SIZE = 32

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'train',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    'test',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Compute class weights to handle imbalance
classes = list(train_generator.class_indices.keys())
y_integers = train_generator.classes
class_weights_values = compute_class_weight(class_weight='balanced',
                                            classes=np.unique(y_integers),
                                            y=y_integers)
class_weights = dict(enumerate(class_weights_values))

# ============================================================
#                   Build MobileNetV2 Model
# ============================================================

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(96, 96, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(7, activation='softmax')(x)  # 7 emotions

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze base layers first
for layer in base_model.layers:
    layer.trainable = False

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ============================================================
#                     Train the Model
# ============================================================

print("=" * 50)
print("  EMOTION BASED MOVIE RECOMMENDER - TRAINING")
print("=" * 50)

history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=25,
    class_weight=class_weights
)

# Fine-tuning
for layer in base_model.layers[-30:]:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history_fine = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=15,
    class_weight=class_weights
)

# ============================================================
#                     Evaluation Metrics
# ============================================================

y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

print("Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=classes))

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred_classes))

# Save model
model.save('emotion_model.h5')
print("Model saved as emotion_model.h5")