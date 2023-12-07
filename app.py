import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline



# Function to load and preprocess the dataset
def load_and_preprocess_data():
    data = pd.read_csv('C:/Users/v-asaha/PycharmProjects/Flight_Delay/Delay.csv')

    data['FLT_ORIG_DT'] = pd.to_datetime(data['FLT_ORIG_DT'])
    data['Year'] = data['FLT_ORIG_DT'].dt.year
    data['Month'] = data['FLT_ORIG_DT'].dt.month
    data['Day'] = data['FLT_ORIG_DT'].dt.day
    data = data.drop(columns=['FLT_ORIG_DT'])

    categorical_columns = ['ACT_ORIG', 'ACT_DEST', 'OPER_CARR', 'DELAY_REASON']
    numeric_columns = ['Year', 'Month', 'Day', 'FLT_NBR']
    target_column = 'DELAY_MIN'

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_columns),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
        ])

    X = data[numeric_columns + categorical_columns]
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, preprocessor


# Function to train the model with hyperparameter tuning
def train_model(X_train, y_train, preprocessor):
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42))
    ])

    param_grid = {
        'regressor__n_estimators': [50, 100, 200],
        'regressor__max_depth': [None, 10, 20],
        'regressor__min_samples_split': [2, 5, 10],
        'regressor__min_samples_leaf': [1, 2, 4]
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='r2')
    grid_search.fit(X_train, y_train)

    return grid_search


# Function to make predictions and evaluate the model
def evaluate_model(grid_search, X_test, y_test):
    predictions = grid_search.predict(X_test)
    r2 = r2_score(y_test, predictions)

    return predictions, r2


# Function to plot histograms, PDFs, and CDFs of residuals
def plot_residuals(residuals):
    plt.figure(figsize=(12, 6))
    plt.hist(residuals, bins=30, density=True, alpha=0.5, color='green', edgecolor='black')
    plt.title('Histogram of Residuals')
    plt.xlabel('Residuals (Actual Delay - Predicted Delay)')
    plt.ylabel('Probability Density')
    plt.grid(True)
    plt.show()

    pdf, bins, _ = plt.hist(residuals, bins=30, density=True, alpha=0.5, color='green', edgecolor='black')
    plt.close()

    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    plt.figure(figsize=(12, 6))
    plt.plot(bin_centers, pdf, label='PDF', color='green')
    plt.title('Probability Density Function (PDF) of Residuals')
    plt.xlabel('Residuals (Actual Delay - Predicted Delay)')
    plt.ylabel('Probability Density')
    plt.grid(True)
    plt.show()

    cdf = np.cumsum(pdf) * (bins[1] - bins[0])

    plt.figure(figsize=(12, 6))
    plt.plot(bin_centers, cdf, label='CDF', color='orange')
    plt.title('Cumulative Distribution Function (CDF) of Residuals')
    plt.xlabel('Residuals (Actual Delay - Predicted Delay)')
    plt.ylabel('Cumulative Probability')
    plt.grid(True)
    plt.show()


# Function to predict delays for all origins and routes for the next 10 days
# Function to predict delays for all origins and routes for the next 10 days
def predict_future_delays(data, grid_search):
    all_origins = data['ACT_ORIG'].unique()
    all_destinations = data['ACT_DEST'].unique()

    predicted_data_origin_route = []

    for origin in all_origins:
        for destination in all_destinations:
            filtered_data = data[(data['ACT_ORIG'] == origin) & (data['ACT_DEST'] == destination)]

            if not filtered_data.empty:
                new_data_route = pd.DataFrame({
                    'Year': [2023] * 10, 'Month': [12] * 10, 'Day': range(20, 30), 'FLT_NBR': [1] * 10,
                    'ACT_ORIG': [origin] * 10, 'ACT_DEST': [destination] * 10,
                    'OPER_CARR': ['Carrier'] * 10, 'DELAY_REASON': ['Reason'] * 10
                })

                delay_predictions_route = grid_search.predict(new_data_route)
                mean_delay = np.mean(delay_predictions_route)

                # Check if there are rows in filtered_data before calculating mode
                if not filtered_data['DELAY_REASON'].empty:
                    predicted_delay_reason = filtered_data['DELAY_REASON'].mode().values[0]
                else:
                    predicted_delay_reason = 'Unknown'

                max_delay_date = new_data_route.loc[delay_predictions_route.argmax(), 'Day']
                max_delay_route = f"{origin} to {destination}"

                predicted_data_origin_route.append({
                    'Origin': origin,
                    'Destination': destination,
                    'Mean Predicted Delay': mean_delay,
                    'Predicted Delay Reason': predicted_delay_reason,
                    'Max Delay Date': max_delay_date,
                    'Max Delay Route': max_delay_route
                })

    predicted_df_origin_route = pd.DataFrame(predicted_data_origin_route)

    most_delayed_origin = predicted_df_origin_route.loc[predicted_df_origin_route['Mean Predicted Delay'].idxmax()]

    return most_delayed_origin, predicted_df_origin_route



# Function to save predicted delays to an Excel file
def save_to_excel(predicted_df_origin_route):
    predicted_df_origin_route.to_excel('predicted_delays_origin_route_table.xlsx', index=False)


# Main function to handle user queries
def main():
    X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data()
    grid_search = train_model(X_train, y_train, preprocessor)
    predictions, r2 = evaluate_model(grid_search, X_test, y_test)
    residuals = y_test - predictions

    # Load the data here
    data = pd.read_csv('C:/Users/v-asaha/PycharmProjects/Flight_Delay/Delay.csv')

    while True:
        query = input("Ask me a question (type 'exit' to end): ")

        if query.lower() == 'exit':
            break
        elif 'train model' in query.lower():
            print("Training the model...")
            grid_search = train_model(X_train, y_train, preprocessor)
            print("Model trained successfully!")
        elif 'evaluate model' in query.lower():
            print(f'R2 Score: {r2}')
            plot_residuals(residuals)
        elif 'predict delays' in query.lower():
            most_delayed_origin, predicted_df_origin_route = predict_future_delays(data, grid_search)
            print("Most Delayed Origin:")
            print(most_delayed_origin)
            save_to_excel(predicted_df_origin_route)
            print("Predicted delays for all origins and routes saved to 'predicted_delays_origin_route_table.xlsx'")
        else:
            print("I'm sorry, I don't understand that question. Please ask something else.")


if __name__ == "__main__":
    main()