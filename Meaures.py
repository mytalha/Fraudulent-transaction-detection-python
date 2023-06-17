import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression

# Load your dataset
data1 = pd.read_csv("C:\\Users\\Malik\\Desktop\\fraudtest\\fraudTest.csv")
data2 = pd.read_csv("C:\\Users\\Malik\\Desktop\\fraudtest\\fraudTrain.csv")

# concatanating the two data sets together

data = pd.concat([data1, data2])
data


# taking only the first 100,000 data points to make it easier to do computations on

data = data.sample(frac=1, random_state=1).reset_index()
data = data.head(n=100000)
data.is_fraud.value_counts()


# doing some prelimenary adjustments to data set
data = data.drop('Unnamed: 0', axis=1)
data = data.drop('trans_num', axis=1)
data = data.drop('first', axis=1)
data = data.drop('last', axis=1)
data = data.drop('street', axis=1)
data = data.drop('city', axis=1)


# calculating distance between credit card holder location and location of merchant

data['distance'] = np.sqrt((data['lat'] - data['merch_lat'])**2 + (data['long'] - data['merch_long'])**2)

# converting to date time
data['trans_date_trans_time'] = pd.to_datetime(data['trans_date_trans_time'])

# pulling the hour for a variable
def pull_hour(ts):
    return ts.hour
data['hour'] = data['trans_date_trans_time'].apply(pull_hour)



# using unix time we are going to calculate the sum of transaction amounts in past 30 days
# we will create two variables from this, first sum of transactions in 30 days
# next will be a interaction based variable between current purchase amount / 30 day total
# this will help measure if this transaction is out of the ordinary



# function to calculate last 30 day spending
def sum_30_day(unixtime, cc_num):
    unixstamp = unixtime
    minus30 = unixstamp - 2629743
    ccnum = cc_num
    sumtable = data.loc[(data["cc_num"] == ccnum) & (data['unix_time'] < unixstamp) & (data['unix_time'] > minus30)]
    history30 = sumtable['amt'].sum()
    return history30

    
# running function and creating a new variable for it
data['history_30'] = data.apply(lambda x: sum_30_day(x.unix_time, x.cc_num), axis=1)


# measuring interaction effect with amt in new variable
data['interaction_30'] = data['history_30'] / data['amt']



# dropping non categorical variables in preperation for regression modeling

data = data.drop('trans_date_trans_time', axis=1)
data = data.drop('state', axis=1)
data = data.drop('merchant', axis=1)
data = data.drop('job', axis=1)
data = data.drop('dob', axis=1)
data = data.drop('category', axis=1)
data = data.drop('gender', axis=1)
data = data.drop('index', axis=1)

# creating a correlation heatmap
# using this we will check for any multicollinearity issues
# multicollinearity is when two variables have a correlation >0.7 with eachother


fig, ax = plt.subplots(figsize=(20,10))
sns.heatmap(data.corr(),annot=True).set_title('Correlation')


# there is multicollinearity issues with our non generated predictors such as lat longs, we will drop all of these

data = data.drop('cc_num', axis=1)
data = data.drop('zip', axis=1)
data = data.drop('lat', axis=1)
data = data.drop('long', axis=1)
data = data.drop('unix_time', axis=1)
data = data.drop('merch_lat', axis=1)
data = data.drop('merch_long', axis=1)



fig, ax = plt.subplots(figsize=(20,10))
sns.heatmap(data.corr(),annot=True).set_title('Correlation')
# Split the data into training and testing sets
y = data['is_fraud']
x = data.drop('is_fraud', axis=1) 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# Fit the logistic regression model
model = LogisticRegression()
model.fit(x_train, y_train)

# Predict the labels for the test set
y_pred = model.predict(x_test)

# Calculate accuracy, precision, recall, and F1-score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print the results
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
