# Luisa Rosa
# HW#2 - Machine Learning in Finance
# Question 6

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense
from sklearn.metrics import mean_squared_error, r2_score

# Create lagged dataset with 50 time steps
def create_lagged_dataset(data, original_index, lag=50):
    X, y, dates = [], [], []
    for i in range(lag, len(data)):
        X.append(data[i-lag:i, 0])
        y.append(data[i, 0])
        dates.append(original_index[i])
    return np.array(X), np.array(y), dates

# Build Model
def buildModel(model_name, epochs, learn_rate):
    model = Sequential()
    model.add(model_name(epochs, activation='tanh', input_shape=(X_train.shape[1], 1))) # 'tahn' helps add non-linearity
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer=SGD(learning_rate=learn_rate, momentum=0.9), loss='mse') # momentum helps the optimizer navigate the loss more efficiently

    # Train the model
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True) # helps prevent overfitting and saves training time
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=1, callbacks=[early_stopping])

    # Make predictions
    y_pred = model.predict(X_test)

    # Inverse transform the predictions and actual values
    y_pred_inverse = scaler.inverse_transform(y_pred)
    y_test_inverse = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Calculate MSE and R2 for the test data
    test_mse = mean_squared_error(y_test_inverse, y_pred_inverse)
    test_r2 = r2_score(y_test_inverse, y_pred_inverse)

    # Print the results
    print("Model Architecture Summary:")
    model.summary()
    print(f"Number of Epochs Trained: {len(history.epoch)}")
    print(f"Final Training MSE Loss: {history.history['loss'][-1]}")
    print(f"Test Data MSE: {test_mse}")
    print(f"Test Data R²: {test_r2}")

    # Output each predicted point and the corresponding forecasted value with date
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

# #get file in colab
# from google.colab import files
# df = files.upload()

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
X_train, y_train, train_dates = create_lagged_dataset(train_scaled, train_data.index)
X_test, y_test, test_dates = create_lagged_dataset(test_scaled, test_data.index)

# Reshape input to be [samples, time steps, features]
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Build the RNN model
rnn_y_test, rnn_y_pred, rnn_hist = buildModel(SimpleRNN, 100, 0.01)
visualize("RNN", rnn_y_test, rnn_y_pred, rnn_hist)

# Build the LSTM model
lstm_y_test, lstm_y_pred, lstm_hist = buildModel(LSTM, 100, 0.1)
visualize("LSTM", lstm_y_test, lstm_y_pred, lstm_hist)

    


