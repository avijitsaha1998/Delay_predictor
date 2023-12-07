import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
# Load your flight delay data from Excel
excel_file_path = 'C:/Users/v-asaha/PycharmProjects/Flight_Delay/Delay.xlsx'
df = pd.read_excel(excel_file_path)
# Select relevant columns
selected_columns = ['ACT_ORIG', 'ACT_DEST', 'FLT_NBR', 'OPER_CARR', 'DELAY_REASON', 'DELAY_MIN']
df = df[selected_columns]
# Handle categorical columns using one-hot encoding
categorical_columns = ['ACT_ORIG', 'ACT_DEST', 'OPER_CARR', 'DELAY_REASON']
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
# Handle non-numeric values in numeric columns
numeric_columns = ['FLT_NBR']
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
# Drop rows with missing values
df.dropna(inplace=True)
# Define features (X) and target variable (y)
X = df.drop('DELAY_MIN', axis=1) # Features
y = df['DELAY_MIN'] # Target variable (delay duration)
# Check dataset size
print("Number of samples in the dataset:", len(df))
# Split the data into training and testing sets with a reduced test_size
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train a simple model (Random Forest Regressor in this example)
model = RandomForestRegressor()
model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test)
# Evaluate the model (for regression, you can use metrics like Mean Squared Error)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
# Save the trained model to a file
model_filename = 'flight_delay_model.joblib'
joblib.dump(model, model_filename)
print(f'Trained model saved to {model_filename}')