import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load your dataset
data = pd.read_csv('C:/Users/v-asaha/PycharmProjects/Flight_Delay/Delay.csv')

# Preprocess the data
# Convert date to datetime format
data['FLT_ORIG_DT'] = pd.to_datetime(data['FLT_ORIG_DT'])
# Extract day, month, and year from the date
data['Year'] = data['FLT_ORIG_DT'].dt.year
data['Month'] = data['FLT_ORIG_DT'].dt.month
data['Day'] = data['FLT_ORIG_DT'].dt.day
# Drop the original date column
data = data.drop(columns=['FLT_ORIG_DT'])

# Encoding categorical variables using one-hot encoding
categorical_columns = ['ACT_ORIG', 'ACT_DEST', 'OPER_CARR', 'DELAY_REASON']
numeric_columns = ['Year', 'Month', 'Day', 'FLT_NBR']
target_column = 'DELAY_MIN'

# Creating a ColumnTransformer to apply different preprocessing steps to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
    ])

# Define the feature columns and target column
X = data[numeric_columns + categorical_columns]
y = data[target_column]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with preprocessing and Random Forest regressor
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Define hyperparameters for tuning
param_grid = {
    'regressor__n_estimators': [50, 100, 200],
    'regressor__max_depth': [None, 10, 20],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 4]
}

# Use GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='r2')
grid_search.fit(X_train, y_train)

# Print the best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)

# Make predictions on the test set
predictions = grid_search.predict(X_test)

# Evaluate the performance of the model
r2 = r2_score(y_test, predictions)
print(f'R2 Score: {r2}')

# Calculate residuals
residuals = y_test - predictions

# Plot histogram of residuals
plt.figure(figsize=(12, 6))
plt.hist(residuals, bins=30, density=True, alpha=0.5, color='green', edgecolor='black')
plt.title('Histogram of Residuals')
plt.xlabel('Residuals (Actual Delay - Predicted Delay)')
plt.ylabel('Probability Density')
plt.grid(True)
plt.show()

# Calculate PDF
pdf, bins, _ = plt.hist(residuals, bins=30, density=True, alpha=0.5, color='green', edgecolor='black')
plt.close()

# Calculate bin centers
bin_centers = 0.5 * (bins[1:] + bins[:-1])

# Plot PDF
plt.figure(figsize=(12, 6))
plt.plot(bin_centers, pdf, label='PDF', color='green')
plt.title('Probability Density Function (PDF) of Residuals')
plt.xlabel('Residuals (Actual Delay - Predicted Delay)')
plt.ylabel('Probability Density')
plt.grid(True)
plt.show()

# Calculate CDF
cdf = np.cumsum(pdf) * (bins[1] - bins[0])

# Plot CDF
plt.figure(figsize=(12, 6))
plt.plot(bin_centers, cdf, label='CDF', color='orange')
plt.title('Cumulative Distribution Function (CDF) of Residuals')
plt.xlabel('Residuals (Actual Delay - Predicted Delay)')
plt.ylabel('Cumulative Probability')
plt.grid(True)
plt.show()

# Predict delays for all flight numbers
all_flight_numbers = data['FLT_NBR'].unique()
all_flight_numbers.sort()
predicted_data = []

for flight_number in all_flight_numbers:
    # Create new data with the specific flight number
    new_data_flight = pd.DataFrame({
        'Year': [2023], 'Month': [12], 'Day': [19], 'FLT_NBR': [flight_number],
        'ACT_ORIG': ['Origin'], 'ACT_DEST': ['Destination'],
        'OPER_CARR': ['Carrier'], 'DELAY_REASON': ['Reason']
    })

    # Predict delay for the specific flight number
    delay_prediction = grid_search.predict(new_data_flight)

    # Get the corresponding data for the flight number
    flight_data = data[data['FLT_NBR'] == flight_number].iloc[0]

    # Append the results to the list
    predicted_data.append({
        'Flight Number': flight_number,
        'Predicted Delay': delay_prediction[0],
        'Origin': flight_data['ACT_ORIG'],
        'Destination': flight_data['ACT_DEST'],
        'Delay Reason': flight_data['DELAY_REASON']
    })

# Create a DataFrame from the list
predicted_df = pd.DataFrame(predicted_data)

# Plotting the results
plt.figure(figsize=(12, 6))
plt.scatter(all_flight_numbers, predicted_df['Predicted Delay'], label='Predicted Delays', color='blue', marker='o')
plt.title('Predicted Delays for Different Flight Numbers')
plt.xlabel('Flight Number')
plt.ylabel('Predicted Delay (minutes)')
plt.legend()
plt.grid(True)
plt.show()

# Save the table to an Excel file
predicted_df.to_excel('predicted_delays_table.xlsx', index=False)

#Sorting Delay prone routes and an origin
#Chatbot based on data where for a flight the next 15 days calculating the delay probability