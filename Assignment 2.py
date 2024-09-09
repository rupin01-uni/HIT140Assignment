# Import the required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets (ensure these are in your working directory or specify the correct path)
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

# Verify that columns are correctly renamed and present
print("Columns in the dataset after renaming:")
print(merged_data.columns)

# Plot screen time distribution by gender using a bar chart for Computer usage on weekdays
plt.figure(figsize=(10, 6))
sns.histplot(data=merged_data, x='Computer_Weekdays', hue='Gender', multiple='stack', bins=20)
plt.title('Distribution of Computer Usage on Weekdays by Gender')
plt.xlabel('Hours per Day')
plt.ylabel('Frequency')
plt.show()

# Bar chart for smartphone usage on weekends by minority status
plt.figure(figsize=(10, 6))
sns.histplot(data=merged_data, x='Smartphone_Weekends', hue='Minority_Status', multiple='stack', bins=20)
plt.title('Distribution of Smartphone Usage on Weekends by Minority Status')
plt.xlabel('Hours per Day')
plt.ylabel('Frequency')
plt.show()

# Bar chart for TV usage on weekdays by deprivation status
plt.figure(figsize=(10, 6))
sns.histplot(data=merged_data, x='TV_Weekdays', hue='Deprivation_Status', multiple='stack', bins=20)
plt.title('Distribution of TV Usage on Weekdays by Deprivation Status')
plt.xlabel('Hours per Day')
plt.ylabel('Frequency')
plt.show()

# Define well-being columns to plot
well_being_columns = ['Optimistic', 'Feeling_Useful', 'Feeling_Relaxed', 'Interested_in_People', 
                      'Energy_to_Spare', 'Dealing_with_Problems', 'Thinking_Clearly', 
                      'Feeling_Good_about_Myself', 'Feeling_Close_to_Others', 'Feeling_Confident', 
                      'Making_My_Own_Mind', 'Feeling_Loved', 'Interested_in_New_Things', 'Feeling_Cheerful']

# Ensure the columns exist in the DataFrame
existing_well_being_columns = [col for col in well_being_columns if col in merged_data.columns]

# Plot histograms for well-being indicators
for column in existing_well_being_columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(data=merged_data, x=column, bins=5)
    plt.title(f'Distribution of {column}')
    plt.xlabel('Score (1-5)')
    plt.ylabel('Frequency')
    plt.show()

# Select screen time and well-being columns
screen_time_columns = ['Computer_Weekends', 'Computer_Weekdays', 'Gaming_Weekends', 'Gaming_Weekdays', 
                       'Smartphone_Weekends', 'Smartphone_Weekdays', 'TV_Weekends', 'TV_Weekdays']

# Combine screen time and well-being columns for correlation analysis
correlation_columns = screen_time_columns + existing_well_being_columns
correlation_data = merged_data[correlation_columns]

# Compute the correlation matrix
correlation_matrix = correlation_data.corr()

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Screen Time and Well-Being Indicators')
plt.show()
