# Part 1: Analyzing the Data

# a) Loading and Exploring the Data

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime

current_time = datetime.datetime.now() # () znamenají, že to teoreticky bere info z PC, kde jsem
current_time = current_time.strftime("%Y_%m_%d_%H_%M_%S")
print(current_time)

# Load the dataset
url = "https://raw.githubusercontent.com/MartaCap/2024_ells_python/main/thursday/competition/wine_quality.csv"
df = pd.read_csv(url)

# Display the first few rows of the dataset
print(df.head())

#b Data Exploraton

# Display dataset info
print(df.info())

# Check for unique values
print(df.nunique())

# Check for missing values
print(df.isnull().sum())

# Check for duplicated rows
print(df.duplicated().sum())

# Remove missing values and duplicated rows if any
df = df.dropna().drop_duplicates()

# c)  Data exploration

# Display dataset info
print(df.info())

# Check for unique values
print(df.nunique())

# Check for missing values
print(df.isnull().sum())

# Check for duplicated rows
print(df.duplicated().sum())

# Remove missing values and duplicated rows if any
df = df.dropna().drop_duplicates()

# d) Analysis of Categorical Variables

# Subset for categorical variables (only 'quality')
categorical_vars = df["quality"]
print(categorical_vars.head(), categorical_vars.shape)

# Unique values and summary statistics
print(categorical_vars.nunique())
print(categorical_vars.describe())

# Count plot for 'quality'
plt.figure(figsize=(10, 6))
sns.countplot(x="quality", data=df)
plt.title("Count_Plot_for_Quality".replace(" ", "_")) # nahrazeni mezer podtrzitkem
plt.savefig(f"./Pictures/Count_plot_for_quality_{current_time}.png")
#plt.show()

# Insights
value_counts = df["quality"].value_counts()
print(value_counts)
balanced = value_counts.max() / value_counts.min() < 1.5
balance_status = "balanced" if balanced else "not balanced"
print(f"The dataset is {balance_status} with respect to the 'quality' categories.")

# e) Analysis of Numerical Variables

# Subset for numerical variables
numerical_vars = df.select_dtypes(include=["float64", "int64"])

# Summary statistics
summary_stats_numerical = numerical_vars.describe().T
summary_stats_numerical["variance"] = numerical_vars.var()
summary_stats_numerical["std"] = numerical_vars.std()
print(summary_stats_numerical)

# Graphical analysis of numerical variables
for col in numerical_vars.columns:
    plt.figure(figsize=(14, 6))
    
    # Histogram
    plt.subplot(1, 3, 1)
    sns.histplot(numerical_vars[col], kde=True)
    plt.title(f"Histogram_of_{col}")
    plt.savefig(f"./Pictures/Histogram_of_{col}_{current_time}.png")

    
    # Box plot
    plt.subplot(1, 3, 2)
    sns.boxplot(y=numerical_vars[col])
    plt.title(f"Box_Plot_of_{col}")
    plt.savefig(f"./Pictures/Box_Plot_of_{col}_{current_time}.png")


    # Violin plot
    plt.subplot(1, 3, 3)
    sns.violinplot(y=numerical_vars[col])
    plt.title(f"Violin_Plot_of_{col}")
    
    plt.tight_layout()
    plt.savefig(f"./Pictures/tight_layout_{current_time}.png")
    #plt.show()

# Relationships between numerical variables
sns.pairplot(numerical_vars)
plt.savefig(f"./Pictures/numerical_vars_{current_time}.png")
# plt.show()

# Correlation matrix
correlation_matrix = numerical_vars.corr()
print(correlation_matrix)

# Heatmap of the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation_Matrix_of_Numerical_Variables")
plt.savefig(f"./Pictures/Correlation_Matrix_of_Numerical_Variables_{current_time}.png")
# plt.show()

# Part 2: Creating a Perceptron Model

# a) Data Preparation

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score

# Convert "quality" to binary classification
df["quality"] = df["quality"].apply(lambda x: 1 if x == "good wine" else 0)
print(df.head())

# Separate features and target
X = df.drop("quality", axis=1)
y = df["quality"]

# Handle class imbalance with SMOTE
oversample = SMOTE(k_neighbors=5)
X_ros, y_ros = oversample.fit_resample(X, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_ros, y_ros, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Perceptron model
clf = Perceptron(eta0=0.1, max_iter=1000, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# b) Model Evaluation

# Evaluation with repeated train-test split
import numpy as np
import statistics

num_repetitions = 1000
accuracy_scores = []

for repetition in range(num_repetitions):
    X_train, X_test, y_train, y_test = train_test_split(X_ros, y_ros, test_size=0.2, random_state=42 + repetition)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy_scores.append(accuracy_score(y_test, y_pred))

print("Min. Accuracy Score:", min(accuracy_scores))
print("Max. Accuracy Score:", max(accuracy_scores))
print("Mean Accuracy Score:", statistics.mean(accuracy_scores))
print("Median Accuracy Score:", statistics.median(accuracy_scores))

plt.plot(range(num_repetitions), accuracy_scores)
plt.xlabel("Repetition")
plt.ylabel("Accuracy_Score")
plt.title("Model_Accuracy_over_1000_Repetitions")
plt.savefig(f"./Pictures/Model_Accuracy_over_1000_Repetitions_{current_time}.png")
# plt.show()