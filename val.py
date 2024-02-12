import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = {
    'Year': [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],
    'PercentCelebrating': [60, 58, 59, 60, 54, 55, 55, 54, 55, 51, 55, 52, 53],
    'PerPerson': [103.00, 116.21, 126.03, 130.97, 133.91, 142.31, 146.84, 136.57, 143.56, 161.96, 196.31, 164.76, 175.41],
    'Candy': [8.60, 10.75, 10.85, 11.64, 10.80, 12.70, 13.11, 12.68, 13.12, 14.12, 17.30, 15.32, 15.90],
    'Flowers': [12.33, 12.62, 13.49, 13.48, 15.00, 15.72, 14.78, 14.63, 14.75, 15.07, 16.49, 15.42, 16.71],
    'Jewelry': [21.52, 26.18, 29.60, 30.94, 30.58, 36.30, 33.11, 32.32, 34.10, 30.34, 41.65, 30.71, 45.75],
    'GreetingCards': [5.91, 8.09, 6.93, 8.32, 7.97, 7.87, 8.52, 7.36, 6.55, 7.31, 9.01, 8.48, 7.47],
    'EveningOut': [23.76, 24.86, 25.66, 27.93, 27.48, 27.27, 33.46, 28.46, 26.96, 27.72, 30.78, 21.39, 31.35],
    'Clothing': [10.93, 12.00, 10.42, 11.46, 13.37, 14.72, 15.05, 13.91, 14.04, 16.08, 20.67, 20.05, 21.46],
    'GiftCards': [8.42, 11.21, 8.43, 10.23, 9.00, 11.05, 12.52, 10.23, 11.04, 10.31, 14.21, 15.67, 17.22]
}

# Create DataFrame
df = pd.DataFrame(data)

# Display the first few rows of the DataFrame
print(df.head())

# Prepare the features (independent variable)
X = df[['Year']]

# Prepare the target variables (dependent variables)
y = df[['Candy', 'Flowers', 'Jewelry', 'GreetingCards', 'EveningOut', 'Clothing', 'GiftCards']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)

with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)
