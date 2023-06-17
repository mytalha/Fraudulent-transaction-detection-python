import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression

# Load the dataset
data1 = pd.read_csv("D:\\AIFinalAPP\\fraudTest.csv")
data2 = pd.read_csv("D:\\AIFinalAPP\\fraudTrain.csv")
data = pd.concat([data1, data2])

# Take a random sample of 10,000 data points for training
data_train = data.sample(n=100000, random_state=1)

# Perform preliminary adjustments to the dataset
data_train = data_train.drop(['Unnamed: 0', 'trans_num', 'first', 'last', 'street', 'city'], axis=1)
data_train['distance'] = ((data_train['lat'] - data_train['merch_lat'])**2 + (data_train['long'] - data_train['merch_long'])**2) ** 0.5
data_train['hour'] = pd.to_datetime(data_train['trans_date_trans_time']).dt.hour

# Calculate the sum of transaction amounts in the past 30 days
def sum_30_day(cc_num, unix_time):
    minus30 = unix_time - 2629743
    sumtable = data_train.loc[(data_train['cc_num'] == cc_num) & (data_train['unix_time'] > minus30) & (data_train['unix_time'] < unix_time)]
    return sumtable['amt'].sum()

data_train['history_30'] = data_train.apply(lambda x: sum_30_day(x['cc_num'], x['unix_time']), axis=1)
data_train['interaction_30'] = data_train['history_30'] / data_train['amt']

# Drop non-categorical variables
data_train = data_train.drop(['trans_date_trans_time', 'state', 'merchant', 'job', 'dob', 'category', 'gender'], axis=1)

# Drop variables causing multicollinearity
data_train = data_train.drop(['cc_num', 'zip', 'lat', 'long', 'unix_time', 'merch_lat', 'merch_long'], axis=1)

# Fit the logistic regression model
y_train = data_train['is_fraud']
x_train = data_train.drop('is_fraud', axis=1)
model = LogisticRegression()
model.fit(x_train, y_train)

# Save the trained model
joblib.dump(model, "model.pkl")

# Function to predict fraud based on user input
def predict_fraud(trans_amt, hour, history_30, city_pop, distance):
    interaction_30 = history_30 / trans_amt

    # Create the input data point with correct column order
    data_point = pd.DataFrame({
        'amt': [trans_amt],
        'hour': [hour],
        'history_30': [history_30],
        'interaction_30': [interaction_30],
        'city_pop': [city_pop],
        'distance': [distance]
    }, columns=x_train.columns)

    # Load the trained model
    loaded_model = joblib.load("model.pkl")

    prediction = loaded_model.predict(data_point)[0]
    if prediction == 1:
        return "The transaction is classified as fraudulent."
    else:
        return "The transaction is classified as legitimate."

# # Example usage: predict fraud for user input
# trans_amt = 100.0
# hour = 10
# history_30 = 500.0
# city_pop = 1000000.0
# distance = 5.0

# result = predict_fraud(trans_amt, hour, history_30, city_pop, distance)
# print(result)
