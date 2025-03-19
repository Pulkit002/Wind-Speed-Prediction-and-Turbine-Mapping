import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import os
import time
import joblib
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Load the trained model and scaler
model = load_model(r"model\lstm_wind_speed_model.h5")
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

scaler = joblib.load(r"model\scaler.pkl")

# Load CSV data for simulation
turbine_data = pd.read_csv(r"data\HDGeneratorIREG01.csv") # This is only turbine1 data but in actual we need all 16 turbine data to get their IREG_UBUS, IREG_IBUS, IREG_PWM
data = pd.read_csv(r"sample_wind_prediction_data.csv")  # Replace with your file path
wind_speed_data = data['windspeed'].values  # Replace 'wind_speed' with your column name if different

# Define threshold parameters
mae_threshold = 5
change_threshold = 10

# Lists to store predictions and actual values
predictions = []
actual_values = []

# Function to normalize incoming wind speed data
def normalize_data(data, scaler):
    return scaler.transform(np.array(data).reshape(-1, 1))

# Function to predict the next point in time series
def predict_next_point(model, data, scaler):
    input_data = np.array(data).reshape(1, 60, 1)  # Reshape to model input
    prediction = model.predict(input_data)
    prediction_actual = scaler.inverse_transform(prediction)
    return prediction_actual[0][0]

# Function to monitor model performance using MAE
def monitor_performance(y_true, y_pred, threshold=5):
    mae = mean_absolute_error(y_true, y_pred)
    if mae > threshold:
        print(f"Model MAE {mae} exceeds threshold {threshold}. Retraining required.")
        return True
    return False

# Function to detect sudden changes in wind speed
def detect_sudden_change(recent_data, change_threshold=10):
    if len(recent_data) >= 2:
        change = abs(recent_data[-1] - recent_data[-2])
        if change > change_threshold:
            print(f"Sudden change detected: {change}. Retraining required.")
            return True
    return False

# Function to retrain the LSTM model
def retrain_model(recent_training_data, recent_training_labels):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    
    # Define the LSTM model structure
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=(60, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))

    # Compile and retrain the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(recent_training_data, recent_training_labels, epochs=5, batch_size=64)
    
    # Save the retrained model (overwrite the old model)
    model.save(r'model\lstm_wind_speed_model.h5')
    print("Model retrained and saved.")

# Initialize recent data for sliding window and training data for retraining
recent_data = wind_speed_data[:60].tolist()
recent_training_data = []
recent_training_labels = []

file_path = r"result"
temp_file_path = r"result\temp_predictions.csv"
is_first_write = True

total_error = 0  # Sum of errors
total_predictions = 0

def write_to_temp_file(df, file_path, is_first_write, max_retries=5, delay=1):
    for attempt in range(max_retries):
        try:
            df.to_csv(file_path, mode='a', header=is_first_write, index=False)
            return True  # Success
        except PermissionError:
            print(f"File is locked. Retry attempt {attempt + 1}/{max_retries}...")
            time.sleep(delay)  # Wait before retrying
    print("Max retries reached. Could not write to CSV.")
    return False

# Periodically merge the temporary file with the main file
def merge_files(temp_file_path, file_path):
    try:
        temp_df = pd.read_csv(temp_file_path)
        temp_df.to_csv(file_path, mode='a', header=not os.path.exists(file_path), index=False)
        os.remove(temp_file_path)  # Remove the temp file after merging
        print(f"Merged data from {temp_file_path} to {file_path}")
    except Exception as e:
        print(f"Error merging files: {e}")
        
# Function to predict using 16 XGBoost models
def windMapping(models, IREG_UBUS, IREG_IBUS, IREG_PWM, windspeed):
    input_array = np.array([[IREG_UBUS, IREG_IBUS, IREG_PWM, windspeed]])  # Convert to 2D array for XGBoost
    
    predictions = []
    i=0
    for model in models:
        X_scaled = xgboost_scalars[i][0].transform(input_array)
        pred_scaled=model.predict(X_scaled)
        pred = xgboost_scalars[i][1].inverse_transform([pred_scaled])
        i=i+1
        predictions.append(pred)
        
    return predictions
        
# Load the 16 XGBoost models and respective scalars
xgboost_models = []
xgboost_scalars = []
for i in range(16):
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(f'C:\\Users\\palak\\OneDrive\\Desktop\\Major Project\\windSpeedPredictionAndMapping\\Final\\model\\turbine{i+1}_model.json')
    xgboost_models.append(xgb_model)
    X_scaler = joblib.load(f'C:\\Users\\palak\\OneDrive\\Desktop\\Major Project\\windSpeedPredictionAndMapping\\Final\\model\\turbine{i+1}_model_X_scaler.pkl')
    y_scaler = joblib.load(f'C:\\Users\\palak\\OneDrive\\Desktop\\Major Project\\windSpeedPredictionAndMapping\\Final\\model\\turbine{i+1}_model_y_scaler.pkl')
    xgboost_scalars.append([X_scaler,y_scaler])

# Taking arbitrary values for the remaining 3 parameters since we don't have real time data
IREG_UBUS = turbine_data['IREG_UBUS'].iloc[1000]
IREG_IBUS = turbine_data['IREG_IBUS'].iloc[1000]
IREG_PWM = turbine_data['IREG_PWM'].iloc[1000]

# Iterate through each point starting from the 61st point
for i in range(60, len(wind_speed_data)):
    new_wind_speed = wind_speed_data[i]
    
    # Predict the next wind speed based on the last 60 points
    normalized_recent_data = normalize_data(recent_data, scaler)
    predicted_wind_speed = predict_next_point(model, normalized_recent_data, scaler)
    predictions.append(predicted_wind_speed)
    actual_values.append(new_wind_speed)
    
    print(f'Predicted wind speed: {predicted_wind_speed}, Actual wind speed: {new_wind_speed}')
    
    # Map the predicted windspeed to respective turbine speeds
    mapped_turbine_speeds=windMapping(xgboost_models,IREG_UBUS,IREG_IBUS,IREG_PWM,predicted_wind_speed)
    
    print("Mapped turbine speeds:")
    print(f"Turbine1: {mapped_turbine_speeds[0]} | Turbine2: {mapped_turbine_speeds[1]} | Turbine3: {mapped_turbine_speeds[2]} | Turbine4: {mapped_turbine_speeds[3]}")
    print(f"Turbine5: {mapped_turbine_speeds[4]} | Turbine6: {mapped_turbine_speeds[5]} | Turbine7: {mapped_turbine_speeds[6]} | Turbine8: {mapped_turbine_speeds[7]}")
    print(f"Turbine9: {mapped_turbine_speeds[8]} | Turbine10: {mapped_turbine_speeds[9]} | Turbine11: {mapped_turbine_speeds[10]} | Turbine12: {mapped_turbine_speeds[11]}")
    print(f"Turbine13: {mapped_turbine_speeds[12]} | Turbine14: {mapped_turbine_speeds[13]} | Turbine15: {mapped_turbine_speeds[14]} | Turbine16: {mapped_turbine_speeds[15]}")
    
    
    # Append normalized data and actual values for potential retraining
    if len(recent_training_data) < 60:
        recent_training_data.append(normalize_data(recent_data[:-1], scaler).tolist())
        recent_training_labels.append(new_wind_speed)

    # Monitor performance by comparing actual wind speed to predicted
    if monitor_performance([new_wind_speed], [predicted_wind_speed], threshold=mae_threshold):
        print("Performance degraded. Retraining model...")
        retrain_model(np.array(recent_training_data), np.array(recent_training_labels))
        
    # Detect sudden changes in wind speed
    if detect_sudden_change(recent_data, change_threshold=change_threshold):
        print("Sudden wind speed change detected. Retraining model...")
        retrain_model(np.array(recent_training_data), np.array(recent_training_labels))

    # Update sliding window with the new point
    recent_data.pop(0)
    recent_data.append(new_wind_speed)
    
    
    accuracy = abs(predicted_wind_speed - new_wind_speed ) / new_wind_speed
    total_error += accuracy
    total_predictions += 1
    
    # Calculate cumulative accuracy (average of all previous accuracies)
    cumulative_accuracy = total_error / total_predictions
    
    # Create a DataFrame with the predictions, actual values, and cumulative accuracy
    results_df = pd.DataFrame({
        'Predicted': [predicted_wind_speed],
        'Actual': [new_wind_speed],
        'Cumulative Error': [cumulative_accuracy]  # Add cumulative accuracy
    })
    
    # Write the data to the CSV file
    if write_to_temp_file(results_df, temp_file_path, is_first_write):
        is_first_write = False  # Only write header once

    # Periodically (e.g., every 100 iterations) merge temp file with the main file
    if i % 1000 == 0:
        merge_files(temp_file_path, file_path)
        
window_size = 60
recent_data = wind_speed_data[-window_size:].tolist()  # Start with the last 60 points

# Predict the next point based only on real data
normalized_recent_data = normalize_data(recent_data, scaler)
predicted_wind_speed = predict_next_point(model, normalized_recent_data, scaler)
predictions.append(predicted_wind_speed)

# Print prediction for the next point (e.g., 101st)
print(f'Predicted wind speed for point {len(wind_speed_data) + 1}: {predicted_wind_speed}')

# Update the sliding window with the predicted value (use it only for the next iteration)
recent_data.pop(0)
recent_data.append(predicted_wind_speed)        

# Merge the remaining data if any
merge_files(temp_file_path, file_path)

# Save predictions and actual values to a CSV file for evaluation
# Calculate and print Mean Absolute Error (MAE) for reference
mae = mean_absolute_error(actual_values[:len(actual_values)-1], predictions[:len(actual_values)-1])
print(f"\nMean Absolute Error (MAE) over simulation: {mae}")
