import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load dataset
data = pd.read_csv("train.csv")

# Select features
X = data[["GrLivArea", "BedroomAbvGr", "FullBath"]]

# Target variable
y = data["SalePrice"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, predictions)

print("Mean Absolute Error:", mae)

# Example prediction
sample_house = [[2000, 3, 2]]
predicted_price = model.predict(sample_house)

print("Predicted House Price:", predicted_price[0])