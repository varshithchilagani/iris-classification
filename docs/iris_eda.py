# Iris Dataset - Exploratory Data Analysis 
#
# This script performs exploratory data analysis (EDA) on the classic Iris dataset.
# It loads the data, provides descriptive statistics, checks for missing values,
# and generates various visualizations to understand feature relationships and species distribution.
#


# ## 1. Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA 

# ## 2. Load and Inspect Data
DATASET_PATH = 'Iris.csv' 

print(f"Loading dataset from {DATASET_PATH}...")
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Remove the 'Id' column if it exists.
if 'Id' in df.columns:
    df = df.drop('Id', axis=1)
    print("Dropped 'Id' column.")

# Display basic information and statistics
print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- DataFrame Info ---")
df.info()

print("\n--- Descriptive Statistics ---")
print(df.describe())

print("\n--- Missing Values Check ---")
print(df.isnull().sum())
if df.isnull().sum().sum() == 0:
    print("No missing values found.")
else:
    print("Missing values detected.") 

print("\n--- Species Distribution ---")
print(df['Species'].value_counts())

# Set plot style for visualizations
sns.set(style="whitegrid")

# 3. Univariate and Bivariate Visualizations

print("\n--- Generating Scatter Plots ---")
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='SepalLengthCm', y='SepalWidthCm', hue='Species', palette='Set1', s=70)
plt.title('Sepal Width vs Sepal Length by Species')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='PetalLengthCm', y='PetalWidthCm', hue='Species', palette='Set1', s=70)
plt.title('Petal Width vs Petal Length by Species')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.show()

print("\n--- Generating Pair Plot ---")
# Shows pairwise relationships between all features, colored by species.
sns.pairplot(df, hue='Species', palette='Set1', diag_kind='kde', markers=["o", "s", "D"])
plt.suptitle('Pair Plot of Iris Features by Species', y=1.02)
plt.show()

print("\n--- Generating Box Plots ---")
# Compare feature distributions across different species.
plt.figure(figsize=(14, 10))

plt.subplot(2, 2, 1)
sns.boxplot(x='Species', y='SepalLengthCm', data=df, palette='Set1')
plt.title('Sepal Length Distribution by Species')
plt.xlabel('Species')
plt.ylabel('Sepal Length (cm)')

plt.subplot(2, 2, 2)
sns.boxplot(x='Species', y='SepalWidthCm', data=df, palette='Set1')
plt.title('Sepal Width Distribution by Species')
plt.xlabel('Species')
plt.ylabel('Sepal Width (cm)')

plt.subplot(2, 2, 3)
sns.boxplot(x='Species', y='PetalLengthCm', data=df, palette='Set1')
plt.title('Petal Length Distribution by Species')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')

plt.subplot(2, 2, 4)
sns.boxplot(x='Species', y='PetalWidthCm', data=df, palette='Set1')
plt.title('Petal Width Distribution by Species')
plt.xlabel('Species')
plt.ylabel('Petal Width (cm)')

plt.tight_layout()
plt.show()

# ## 4. Correlation Analysis

print("\n--- Generating Correlation Heatmap ---")
# Select only numeric columns for correlation calculation.
numeric_df = df.select_dtypes(include=np.number)
correlation_matrix = numeric_df.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Features')
plt.show()

# ## 5. Dimensionality Reduction Visualization (PCA)

print("\n--- Generating PCA Plot ---")
# Separate features (X) and target (y) for PCA.
X = df.drop('Species', axis=1)
y = df['Species']

# Scale the features before applying PCA.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA to reduce to 2 components.
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Create a DataFrame for the PCA results.
pca_df = pd.DataFrame(data=X_pca, columns=['Principal Component 1', 'Principal Component 2'])
pca_df['Species'] = y # Add species information for coloring.

# Plot the PCA results.
plt.figure(figsize=(9, 7))
sns.scatterplot(data=pca_df, x='Principal Component 1', y='Principal Component 2', hue='Species', palette='Set1', s=70)
plt.title('PCA: Iris Dataset Reduced to 2 Components')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Species')
plt.show()

# Display explained variance.
print(f"\nExplained variance ratio by PCA components: {pca.explained_variance_ratio_}")
print(f"Total variance explained by 2 components: {pca.explained_variance_ratio_.sum():.4f}")

print("\n--- EDA Complete ---")