# Iris Classification – Exploratory Data Analysis (EDA)

This project explores the famous **Iris dataset**, one of the most well-known datasets in machine learning.  
The goal is to perform **Exploratory Data Analysis (EDA)** to understand feature relationships, visualize distributions, and apply **PCA (Principal Component Analysis)** for dimensionality reduction and better visualization of species separation.

---

## 📘 Project Overview

This project performs a complete exploratory analysis of the Iris dataset, including:
- Data inspection and cleaning  
- Statistical summaries  
- Visual exploration of relationships between features  
- Correlation heatmap and PCA visualization  

The insights gained from this analysis form the foundation for building predictive classification models in future stages.

---

## 🧠 Dataset Information

The dataset contains **150 records** of iris flowers from three species:

| Feature | Description |
|----------|--------------|
| **SepalLengthCm** | Length of the sepal (in cm) |
| **SepalWidthCm**  | Width of the sepal (in cm) |
| **PetalLengthCm** | Length of the petal (in cm) |
| **PetalWidthCm**  | Width of the petal (in cm) |
| **Species** | Class label – *Iris-setosa*, *Iris-versicolor*, *Iris-virginica* |

**Sample Data:**

| SepalLengthCm | SepalWidthCm | PetalLengthCm | PetalWidthCm | Species |
|----------------|---------------|----------------|---------------|-----------|
| 5.1 | 3.5 | 1.4 | 0.2 | Iris-setosa |
| 4.9 | 3.0 | 1.4 | 0.2 | Iris-setosa |
| 4.7 | 3.2 | 1.3 | 0.2 | Iris-setosa |

---

## 🧩 Features of the Script

✅ Loads and cleans the Iris dataset  
✅ Displays key statistics and missing value report  
✅ Visualizes pairwise relationships using **Seaborn pairplots**  
✅ Generates **box plots**, **scatter plots**, and **correlation heatmaps**  
✅ Applies **PCA** to visualize variance and species separability in 2D space  

---

## 📊 Visualizations Generated

- **Scatter plots** for sepal and petal relationships  
- **Pair plots** showing feature interactions across species  
- **Box plots** to compare distributions by species  
- **Heatmap** showing correlations among numeric features  
- **PCA scatter plot** to visualize dimensionality reduction  

---

## 🧰 Technologies Used

| Library | Purpose |
|----------|----------|
| **Pandas** | Data manipulation and analysis |
| **NumPy** | Numerical computations |
| **Matplotlib** | Data visualization |
| **Seaborn** | Statistical plotting |
| **Scikit-learn** | PCA and data scaling |
