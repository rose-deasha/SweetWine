---

# **Sweet Wines Classification Project 🍷**

## **Overview**
Welcome to the **Sweet Wines Classification Project**! This project uses machine learning to classify wines as **sweet (dessert wines)** or **not sweet**, based on their physicochemical properties. By leveraging the **Wine Quality Dataset** and addressing challenges like **class imbalance**, this project delivers a robust Random Forest-based model with high accuracy.

The project includes data preprocessing, feature importance analysis, class imbalance handling with **SMOTE**, and detailed performance visualizations. It’s a great starting point for exploring machine learning in the food and beverage domain.

---

## **Features**
✨ **Core Highlights**:
- **Comprehensive Data Analysis:** Explore key physicochemical properties like acidity, alcohol, and sugar.
- **SMOTE for Class Imbalance:** Effective balancing of the dataset to enhance minority class predictions.
- **Random Forest Model:** A powerful baseline for wine classification.
- **Rich Visualizations:** Gain insights through confusion matrix heatmaps, feature importance, and more.
- **Extendable Codebase:** Designed for scalability, allowing easy integration of new features or models.

---

## **Dataset**
The dataset contains physicochemical test results and sensory quality ratings for red and white wines. It is sourced from the UCI Machine Learning Repository and is publicly available.

### **Details:**
- **Red Wine Dataset:** `winequality-red.csv`
- **White Wine Dataset:** `winequality-white.csv`
- **Size:** 6497 samples (1599 red wines, 4898 white wines)
- **Target Variable:** `is_sweet` (0 = Not Sweet, 1 = Sweet, derived from residual sugar)

### **Input Features:**
1. Fixed Acidity
2. Volatile Acidity
3. Citric Acid
4. **Residual Sugar** (Key indicator for sweetness)
5. Chlorides
6. Free Sulfur Dioxide
7. Total Sulfur Dioxide
8. Density
9. pH
10. Sulphates
11. Alcohol
12. **Wine Type** (Added manually: Red = 1, White = 0)

---

## **Requirements**
Ensure you have Python 3.7+ and the following libraries installed:
- `pandas`
- `numpy`
- `scikit-learn`
- `imbalanced-learn`
- `matplotlib`
- `seaborn`

Install dependencies using:
```bash
pip install -r requirements.txt
```

---

## **How to Run**
Follow these steps to execute the project:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/sweet-wines-classification.git
   ```
2. Navigate to the project folder:
   ```bash
   cd sweet-wines-classification
   ```
3. Run the script:
   ```bash
   python sweetwines.py
   ```

---

## **Visualizations**
This project includes the following visualizations to interpret model performance and data distribution:

1. **Confusion Matrix Heatmap:** Understand true vs. predicted classifications.
2. **Feature Importance Plot:** Identify the most influential features.
3. **Class Distribution Before and After SMOTE:** Observe how SMOTE balances the dataset.
4. **Precision-Recall Curve:** Assess model performance for imbalanced classes.
5. **Residual Sugar Distribution by Sweetness:** Visualize the relationship between sweetness and residual sugar.

Sample visualization (Confusion Matrix Heatmap):

![Confusion Matrix](path/to/your/heatmap-image.png)

---

## **Results**
- **Accuracy:** 100% on the test data.
- **Recall for Sweet Wines:** 80% after balancing with SMOTE.
- **Key Features:** Residual sugar, alcohol content, density, and wine type significantly influence sweetness prediction.

### **Performance Metrics:**
| Metric          | Class 0 (Not Sweet) | Class 1 (Sweet) |
|------------------|---------------------|-----------------|
| **Precision**    | 1.00               | 1.00            |
| **Recall**       | 1.00               | 0.80            |
| **F1-Score**     | 1.00               | 0.89            |

---

## **Future Enhancements**
- **Hyperparameter Tuning:** Optimize Random Forest using techniques like `GridSearchCV`.
- **Model Comparison:** Evaluate alternative models (e.g., Gradient Boosting, XGBoost).
- **Interactive Dashboard:** Deploy with **Streamlit** for real-time predictions.
- **Data Expansion:** Include more wine varieties or external datasets for broader applicability.
- **Threshold Adjustment:** Refine the definition of sweetness for improved classification.

---

## **Contributing**
We welcome contributions! If you’d like to contribute:
- Fork the repository.
- Create a new branch for your feature or bug fix.
- Submit a pull request with detailed explanations of your changes.

---

## **License**
This project is licensed under the [MIT License](LICENSE).

---

## **Contact**
For questions, feedback, or collaboration opportunities, feel free to reach out:
- **GitHub:** [Your GitHub Profile](https://github.com/your-username)

---
