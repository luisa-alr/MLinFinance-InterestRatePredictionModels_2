# Luisa Rosa
# HW#2 - Machine Learning in Finance
# Question 7 - Extra Credit

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Layer, Dense, Dropout, Flatten, Input, LayerNormalization, MultiHeadAttention
from sklearn.metrics import mean_squared_error, r2_score

# #get file in colab
# from google.colab import files
# df = files.upload()

"""
Transformers, unlike Recurrent Neural Networks (RNNs) or Long Short-Term Memory Networks (LSTMs), do not have any built-in notion of order in the data. 
However, Transformers process the entire sequence simultaneously, which means they do not know the relative position of each element unless we explicitly add positional information.
Positional encoding is used to provide the model with the temporal structure of the input sequence, allowing the Transformer to understand that the data is sequential.
Positional encoding introduces information about the positions of each element in a sequence by adding a fixed, deterministic pattern to each vector in the input sequence.
"""

# Positional Encoding Layer
# Understand the sequential nature of the time-series data
class PositionalEncodingLayer(Layer):
    def __init__(self, position, d_model):
        super(PositionalEncodingLayer, self).__init__()
        # Calculate positional encoding matrix for all positions up to 'position' and embedding dimension 'd_model'
        self.pos_encoding = positional_encoding(position, d_model)

    def call(self, inputs):
      # Add positional encoding to the input to provide sequence information
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

# Build positional encoding
def positional_encoding(position, d_model):
  # Create an array representing angles for each position and dimension
    angle_rads = np.arange(position)[:, np.newaxis] / np.power(10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / np.float32(d_model))

    # Apply sine function to even indices (0, 2, 4, ...) in the dimension
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # Apply cosine function to odd indices (1, 3, 5, ...) in the dimension
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    # Add a batch dimension (position, d_model) -> (1, position, d_model) and cast to TensorFlow tensor
    return tf.cast(angle_rads[np.newaxis, ...], dtype=tf.float32)

# Create lagged dataset with 50 time steps
def create_lagged_dataset(data, lag=50):
    X, y = [], []
    for i in range(lag, len(data)):
        X.append(data[i-lag:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# Build Transformer Model with positional encoding and increased depth
def buildTModel():
    # First Layer of Transformer -- Input Layer
    input_layer = Input(shape=(X_train.shape[1], 1))

    # Second Layer -- a Dense layer helps in initial feature extraction by applying a linear transformation followed by a non-linear activation
    # x = Dense(32, activation='tanh')(input_layer) # 'tahn' helps add non-linearity
    x = Dense(32, activation='relu')(input_layer) # 'relu' works better on this dataset

    # Positional encoding is added to provide information about the order of the input sequence
    x = PositionalEncodingLayer(position=X_train.shape[1], d_model=32)(x)

    # Third Layer -- multi-head attention is used to allow the model to attend to multiple parts of the sequence simultaneously
    x = MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
    x = LayerNormalization(epsilon=1e-6)(x)
    x = Dropout(0.2)(x) # used to prevent overfitting

    # Fourth Layer -- multi-head attention again for Increased Depth
    x = MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
    x = LayerNormalization(epsilon=1e-6)(x)
    x = Dropout(0.2)(x) # used to prevent overfitting
    x = Flatten()(x)

    # Fifth Layer -- a Dense layr helps combine features extracted from earlier layers
    # x = Dense(50, activation='tanh')(x) # 'tahn' helps add non-linearity
    x = Dense(50, activation='relu')(x) # 'relu' works better on this dataset
    x = Dropout(0.2)(x) # used to prevent overfitting

    #Sixth Layer -- output layer generates the final prediction
    output_layer = Dense(1)(x) 

    transformer_model = Model(inputs=input_layer, outputs=output_layer)

    # Compile the model with a different loss function
    # Adam's adaptive nature helps deal with these varying gradient magnitudes effectively, making it easier to converge.
    # The Huber loss is less sensitive to outliers than Mean Squared Error (MSE) while being smoother than Mean Absolute Error (MAE).
    transformer_model.compile(optimizer='adam', loss='huber')

    # Train the model
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True) # helps prevent overfitting and saves training time
    history = transformer_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=1, callbacks=[early_stopping])

    # Make predictions
    y_pred = transformer_model.predict(X_test)

    # Inverse transform the predictions and actual values
    y_pred_inverse = scaler.inverse_transform(y_pred)
    y_test_inverse = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Calculate MSE and R2 for the test data
    test_mse = mean_squared_error(y_test_inverse, y_pred_inverse)
    test_r2 = r2_score(y_test_inverse, y_pred_inverse)

    # Print the results
    print("Model Architecture Summary:")
    transformer_model.summary()
    print(f"Number of Epochs Trained: {len(history.epoch)}")
    print(f"Final Training MSE Loss: {history.history['loss'][-1]}")
    print(f"Test Data MSE: {test_mse}")
    print(f"Test Data R²: {test_r2}")

    # Create a DataFrame to output each forecasted value alongside the actual value
    forecast_df = pd.DataFrame({
        'Date': test_data['DATE'].iloc[-len(y_test_inverse):].values,
        'Actual_IR': y_test_inverse.flatten(),
        'Predicted_IR': y_pred_inverse.flatten(),
        'Difference': y_test_inverse.flatten() - y_pred_inverse.flatten()
    })

    print("\nForecasted Values:")
    print(forecast_df.to_string())

    return y_test_inverse, y_pred_inverse, history

# Plot the predictions vs actual values
def visualize(model_name, y_test, y_pred, hist):
    # Plot the Actual vs. Predicted Interest Rate
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='Actual Interest Rate')
    plt.plot(y_pred, label='Predicted Interest Rate')
    plt.title(f'Actual vs. Predicted Interest Rate - {model_name}')
    plt.xlabel('Time Steps')
    plt.ylabel('Interest Rate')
    plt.legend()
    plt.show()

    # Plot the learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(hist.history['loss'], label='Training Loss')
    plt.plot(hist.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Learning Curve')
    plt.legend()
    plt.show()
    
# Load the data
df = pd.read_csv("DGS10.csv")
df["DATE"] = pd.to_datetime(df["DATE"])  # fix data datetime format
df["IR"] = pd.to_numeric(df["IR"], errors="coerce")  # fix missing numerical values
df.dropna(subset=["IR"], inplace=True)  # drop rows with missing numerical values

# Separate train and test datasets
train_data = df[df["DATE"] < "2021-01-01"]
test_data = df[df["DATE"] >= "2021-01-01"]

# Prepare the data for training
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_data[['IR']])
test_scaled = scaler.transform(test_data[['IR']])

# Create lagged dataset with 50 time steps
X_train, y_train = create_lagged_dataset(train_scaled)
X_test, y_test = create_lagged_dataset(test_scaled)

# Reshape input to be [samples, time steps, features]
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Build the Transformer model
t_y_test, t_y_pred, t_hist = buildTModel()
visualize("Transformer", t_y_test, t_y_pred, t_hist)