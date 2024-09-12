import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

# Load datasets (replace with your actual file paths)
dataset1 = pd.read_csv('dataset1.csv')
dataset2 = pd.read_csv('dataset2.csv')
dataset3 = pd.read_csv('dataset3.csv')

# Merge datasets on 'ID'
data = pd.merge(pd.merge(dataset1, dataset2, on='ID'), dataset3, on='ID')

# Print column names to identify any discrepancies
print("Columns in the dataset:")
print(data.columns)

# Calculate total screen time by summing all screen time variables (weekdays and weekends for computers, video games, smartphones, TV)
data['total_screen_time'] = data[['C_we', 'C_wk',
                                    'G_we', 'G_wk',
                                    'S_we', 'S_wk',
                                    'T_we', 'T_wk']].sum(axis=1)

# Well-being columns based on dataset
well_being_columns = ['Optm', 'Relx', 'Intp', 'Engs', 'Dealpr',
                      'Thkclr', 'Goodme', 'Clsep', 'Conf',
                      'Mkmind', 'Loved', 'Intthg', 'Cheer']

# Check if well-being columns exist in the dataset
existing_columns = [col for col in well_being_columns if col in data.columns]

# 1. Histogram for Smartphone Usage on Weekends
plt.figure(figsize=(8, 6))
sns.histplot(data['S_wk'], bins=10, kde=True)
plt.title('Histogram: Smartphone Usage on Weekends')
plt.xlabel('Smartphone Usage (Hours)')
plt.ylabel('Frequency')
plt.show()

# 2. Linear Regression Plot for Computer Usage on Weekdays vs. Feeling Confident
plt.figure(figsize=(8, 6))
sns.regplot(x='C_we', y='Conf', data=data, line_kws={"color": "red"})
plt.title('Regression Plot: Computer Usage on Weekdays vs. Feeling Confident')
plt.xlabel('Computer Usage on Weekdays (Hours)')
plt.ylabel('Feeling Confident Score')
plt.show()

# 3. Linear Regression Plot for TV Usage vs. Feeling Good About Myself
plt.figure(figsize=(8, 6))
sns.regplot(x='T_wk', y='Goodme', data=data, line_kws={"color": "red"})
plt.title('Regression Plot: TV Usage vs. Feeling Good About Myself')
plt.xlabel('TV Usage (Hours)')
plt.ylabel('Feeling Good About Myself Score')
plt.show()

# 4. Box Plot for Smartphone Usage Across Deprivation Status
plt.figure(figsize=(8, 6))
sns.boxplot(x='deprived', y='S_wk', data=data)
plt.title('Box Plot: Smartphone Usage Across Deprivation Status')
plt.xlabel('Deprivation Status')
plt.ylabel('Smartphone Usage on Weekends (Hours)')
plt.show()

# 5. Correlation Matrix
correlation_matrix = data[['total_screen_time'] + existing_columns].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix: Total Screen Time vs Well-Being Indicators')
plt.show()

# 6. Confusion Matrix
# Create a binary target variable based on the confidence score (e.g., above average)
data['Conf_above_avg'] = (data['Conf'] > data['Conf'].median()).astype(int)

# Features for classification
X = data[['total_screen_time', 'C_we', 'T_wk']]  # Example features
y = data['Conf_above_avg']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit a logistic regression model
log_model = LogisticRegression()
log_model.fit(X_train, y_train)

# Make predictions
y_pred = log_model.predict(X_test)

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Display confusion matrix
plt.figure(figsize=(8, 6))
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Below Average Confidence', 'Above Average Confidence']).plot(cmap='Blues')
plt.title('Confusion Matrix: Confidence Score Prediction')
plt.show()

# 7. T-Test for Differences in Smartphone Usage by Deprivation Status
low_deprivation = data[data['deprived'] == 'Low']['S_wk']
high_deprivation = data[data['deprived'] == 'High']['S_wk']

# Perform the T-test
t_stat, p_value = ttest_ind(low_deprivation, high_deprivation)

print(f'T-Test Results: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}')
if p_value < 0.05:
    print("Significant difference in smartphone usage between deprivation statuses.")
else:
    print("No significant difference in smartphone usage between deprivation statuses.")

# (Optional) Display the first few rows of the dataset to understand its structure
print(data.head())
