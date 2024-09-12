import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
@st.cache_data
def load_data():
    file_path = 'dataset-final.csv'  # Update with your correct file path
    data = pd.read_csv(file_path)
    return data

# Preprocessing
def preprocess_data(data):
    # Select relevant features for prediction
    features = ['Employment_demanded', 'Production_(in_Tonnes)', 'Annual_rainfall', 'Yield_(kg/Ha)']
    data = data[features].dropna()  # Drop missing values for simplicity
    return data

# Main function to render the Streamlit app
def main():
    st.title('MGNREGA Employment Demand Prediction')

    # Load the dataset
    data = load_data()

    # Preprocess the data
    data = preprocess_data(data)

    # Feature selection
    X = data[['Production_(in_Tonnes)', 'Annual_rainfall', 'Yield_(kg/Ha)']]
    y = data['Employment_demanded']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model selection: Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Model evaluation
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Display the results
    st.subheader("Model Evaluation")
    st.write(f"Mean Absolute Error: {mae}")
    st.write(f"R-squared: {r2}")

    # Plot actual vs predicted values
    st.subheader("Actual vs Predicted Employment Demand")
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    st.pyplot(fig)

    # Allow user to make custom predictions
    st.subheader("Make Predictions")
    production = st.number_input('Enter Agricultural Production (Tonnes)', value=10000.0)
    rainfall = st.number_input('Enter Annual Rainfall (mm)', value=800.0)
    yield_ = st.number_input('Enter Crop Yield (kg/Ha)', value=2000.0)

    custom_prediction = model.predict([[production, rainfall, yield_]])
    st.write(f"Predicted Employment Demand: {custom_prediction[0]}")

if __name__ == "__main__":
    main()
