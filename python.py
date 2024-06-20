import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.linear_model import LinearRegression

# Define CNN model for food classification
base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(512, activation='relu')(x)
predictions = Dense(num_food_classes, activation='softmax')(x)
food_classification_model = Model(inputs=base_model.input, outputs=predictions)

# Freeze layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Compile the food classification model
food_classification_model.compile(optimizer=Adam(),
                                  loss='categorical_crossentropy',
                                  metrics=['accuracy'])

# Train the food classification model
food_classification_model.fit(train_images, train_labels, epochs=10, validation_data=(val_images, val_labels))

# Extract features for calorie estimation
feature_extractor = Model(inputs=food_classification_model.input,
                          outputs=food_classification_model.get_layer('global_average_pooling2d').output)

train_features = feature_extractor.predict(train_images)
test_features = feature_extractor.predict(test_images)

# Calorie estimation using linear regression
calorie_estimator = LinearRegression()
calorie_estimator.fit(train_features, train_calories)
predicted_calories = calorie_estimator.predict(test_features)

# Evaluate the performance
calorie_mse = mean_squared_error(test_calories, predicted_calories)
print(f'Mean Squared Error for Calorie Estimation: {calorie_mse}')

# Save models for deployment
food_classification_model.save('food_classification_model.h5')
