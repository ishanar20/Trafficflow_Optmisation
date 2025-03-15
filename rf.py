import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

# Load data
data = pd.read_csv('traffic_data.csv')

# Preprocessing
X = data[['RFID', 'Time', 'Speed', 'Volume']]  # Features
y = data['CongestionLevel']  # Target variable

# Label encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Define columns to be one-hot encoded
categorical_cols = ['RFID', 'Time']  # Assuming 'RFID' and 'Time' are categorical

# Create transformer for one-hot encoding with 'ignore' for unknown categories
preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'
)

# Apply preprocessing to training and testing data
X_train_encoded = preprocessor.fit_transform(X_train)
X_test_encoded = preprocessor.transform(X_test)

# Initialize the Random Forest Regressor with encoded data
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model with encoded data
rf_model.fit(X_train_encoded, y_train)

# Make predictions
y_pred = rf_model.predict(X_test_encoded)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Example real-time prediction
new_data = pd.DataFrame({'RFID': ['new_rfid'], 'Time': ['new_time'], 'Speed': [55], 'Volume': [110]})  # Modify 'Speed' and 'Volume' with numeric values
new_data_encoded = preprocessor.transform(new_data)
prediction = rf_model.predict(new_data_encoded)

# Handle unseen labels using predicted label index
predicted_label_idx = int(round(prediction[0]))
if 0 <= predicted_label_idx < len(label_encoder.classes_):
    predicted_label = label_encoder.inverse_transform([predicted_label_idx])[0]
else:
    predicted_label = 'Unknown'

print("Prediction:", prediction)
print("Encoded Labels:", y_encoded)
print("Predicted Label Index:", predicted_label_idx)
print("Predicted Congestion Level:", predicted_label)

# Print label mapping for debugging
label_mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
print("Label Mapping:", label_mapping)
