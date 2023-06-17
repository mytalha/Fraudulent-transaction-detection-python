import matplotlib.pyplot as plt
import pandas as pd

# Read the dataset into a pandas DataFrame
df = pd.read_csv(r'fraudTest.csv')

# Plotting a histogram
plt.hist(df['amt'], bins=10)
plt.xlabel('Amount')
plt.ylabel('Frequency')
plt.title('Histogram of Transaction Amount')
plt.show()

# Plotting a bar chart
category_counts = df['category'].value_counts()
plt.bar(category_counts.index, category_counts.values)
plt.xlabel('Category')
plt.ylabel('Count')
plt.title('Bar Chart of Transaction Categories')
plt.xticks(rotation=90)
plt.show()

# Plotting a scatter plot
plt.scatter(df['merch_lat'], df['merch_long'])
plt.xlabel('Merchant Latitude')
plt.ylabel('Merchant Longitude')
plt.title('Scatter Plot of Merchant Locations')
plt.show()

# Plotting a pie chart
gender_counts = df['gender'].value_counts()
plt.pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%')
plt.title('Pie Chart of Gender Distribution')
plt.show()

# Plotting a line plot
fraud_counts = df.groupby('trans_date_trans_time')['is_fraud'].sum()
plt.plot(fraud_counts.index, fraud_counts.values)
plt.xlabel('Transaction Date and Time')
plt.ylabel('Fraud Count')
plt.title('Line Plot of Fraudulent Transactions over Time')
plt.xticks(rotation=45)
plt.show()
