# Import the required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm  # For linear regression

# Load the datasets
demographics = pd.read_csv('dataset1.csv')  # Demographics data
screen_time = pd.read_csv('dataset2.csv')  # Screen time data
well_being = pd.read_csv('dataset3.csv')  # Well-being data

# Merge datasets on ID
merged_data = demographics.merge(screen_time, on='ID').merge(well_being, on='ID')

# Check for missing values
print("Missing values in the dataset:")
print(merged_data.isnull().sum())

# Rename the columns for better readability
merged_data = merged_data.rename(columns={
    'C_we': 'Computer_Weekends',
    'C_wk': 'Computer_Weekdays',
    'G_we': 'Gaming_Weekends',
    'G_wk': 'Gaming_Weekdays',
    'S_we': 'Smartphone_Weekends',
    'S_wk': 'Smartphone_Weekdays',
    'T_we': 'TV_Weekends',
    'T_wk': 'TV_Weekdays',
    'gender': 'Gender',
    'minority': 'Minority_Status',
    'deprived': 'Deprivation_Status',
    'Optm': 'Optimistic',
    'Usef': 'Feeling_Useful',
    'Relx': 'Feeling_Relaxed',
    'Intp': 'Interested_in_People',
    'Engs': 'Energy_to_Spare',
    'Dealpr': 'Dealing_with_Problems',
    'Thkclr': 'Thinking_Clearly',
    'Goodme': 'Feeling_Good_about_Myself',
    'Clsep': 'Feeling_Close_to_Others',
    'Conf': 'Feeling_Confident',
    'Mkmind': 'Making_My_Own_Mind',
    'Loved': 'Feeling_Loved',
    'Intthg': 'Interested_in_New_Things',
    'Cheer': 'Feeling_Cheerful'
})

# Investigation 1: Histogram for Smartphone Usage on Weekends
plt.figure(figsize=(10, 6))
sns.histplot(data=merged_data, x='Smartphone_Weekends', bins=20)
plt.title('Distribution of Smartphone Usage on Weekends')
plt.xlabel('Hours per Day')
plt.ylabel('Frequency')
plt.show()

# Check correlation between Computer Usage and Feeling Confident
corr1 = merged_data['Computer_Weekdays'].corr(merged_data['Feeling_Confident'])
print(f'Correlation between Computer Usage (Weekdays) and Feeling Confident: {corr1}')

# Investigation 2: Linear Regression for Computer Usage (Weekdays) vs. Feeling Confident
X1 = merged_data[['Computer_Weekdays']]  # Independent variable
y1 = merged_data['Feeling_Confident']    # Dependent variable

X1 = sm.add_constant(X1)  # Add a constant to the model
model1 = sm.OLS(y1, X1).fit()  # Fit the linear regression model
print("Linear Regression Results for Computer Usage vs. Feeling Confident:")
print(model1.summary())  # Display regression results

# Plot the regression line for Computer Usage vs. Feeling Confident with adjusted limits
plt.figure(figsize=(10, 6))
sns.regplot(x='Computer_Weekdays', y='Feeling_Confident', data=merged_data, ci=None, line_kws={"color": "red"})
plt.title('Computer Usage (Weekdays) vs. Feeling Confident')
plt.xlabel('Computer Usage (Weekdays)')
plt.ylabel('Feeling Confident')
plt.xlim(0, merged_data['Computer_Weekdays'].max() + 1)  # Adjust x-axis limit
plt.ylim(0, merged_data['Feeling_Confident'].max() + 1)  # Adjust y-axis limit
plt.show()

# Check correlation between TV Usage and Feeling Good about Myself
corr2 = merged_data['TV_Weekdays'].corr(merged_data['Feeling_Good_about_Myself'])
print(f'Correlation between TV Usage (Weekdays) and Feeling Good about Myself: {corr2}')

# Investigation 3: Linear Regression for TV Usage (Weekdays) vs. Feeling Good about Myself
X2 = merged_data[['TV_Weekdays']]  # Independent variable
y2 = merged_data['Feeling_Good_about_Myself']  # Dependent variable

X2 = sm.add_constant(X2)  # Add a constant to the model
model2 = sm.OLS(y2, X2).fit()  # Fit the linear regression model
print("Linear Regression Results for TV Usage vs. Feeling Good about Myself:")
print(model2.summary())  # Display regression results

# Plot the regression line for TV Usage vs. Feeling Good about Myself with adjusted limits
plt.figure(figsize=(10, 6))
sns.regplot(x='TV_Weekdays', y='Feeling_Good_about_Myself', data=merged_data, ci=None, line_kws={"color": "green"})
plt.title('TV Usage (Weekdays) vs. Feeling Good about Myself')
plt.xlabel('TV Usage (Weekdays)')
plt.ylabel('Feeling Good about Myself')
plt.xlim(0, merged_data['TV_Weekdays'].max() + 1)  # Adjust x-axis limit
plt.ylim(0, merged_data['Feeling_Good_about_Myself'].max() + 1)  # Adjust y-axis limit
plt.show()

# Investigation 4: Box Plot for Smartphone Usage across Deprivation Status
plt.figure(figsize=(10, 6))
sns.boxplot(x='Deprivation_Status', y='Smartphone_Weekends', data=merged_data, palette="Set2")
plt.title('Smartphone Usage on Weekends by Deprivation Status')
plt.xlabel('Deprivation Status')
plt.ylabel('Hours per Day')
plt.xticks([0, 1], ['Non-Deprived', 'Deprived'])  # Rename x-axis ticks
plt.show()
