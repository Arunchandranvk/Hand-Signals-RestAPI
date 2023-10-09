import tensorflow as tf

# Load the model from the saved file
model = tf.keras.models.load_model("D:\Internship Luminar\Ml Api\sign_language-\model\keras_model.h5")

# Compile the loaded model with the desired optimizer, loss, and metrics
model.compile(
    optimizer='adam',  # Specify the optimizer (e.g., 'adam', 'sgd', etc.)
    loss='categorical_crossentropy',  # Specify the loss function
    metrics=['accuracy']  # Specify evaluation metrics
)


# Now, you can use the compiled model for predictions
