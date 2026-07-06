# Titanic - Machine Learning from Disaster 🚢

This is my first complete data science project where I built an end-to-end machine learning pipeline to predict passenger survival on the Titanic. I walked through the entire workflow: reading the data, cleaning it, exploring it visually, handling overfitting, and deploying an ensemble model to get the best result.

## 🛠️ The Workflow

### 1. Data Loading & Cleaning
* Read the raw Titanic dataset using `pandas`.
* Handled missing values (imputed missing ages and filled gaps in essential columns).
* Converted categorical text columns (like `Sex` and `Embarked`) into numerical values so the machine learning algorithms could process them.

### 2. Data Visualization
* Created visual plots to analyze the survival rates across different passenger classes (`Pclass`), age brackets, and genders.
* Used visual distributions to find core hidden patterns before feeding the data to any models.

### 3. Model Building & Hyperparameter Tuning
* **Baseline Tree:** Built an initial Decision Tree that scored 79.89% but heavily overfitted the data by creating too many complex, chaotic branches.
* **Tuned Tree:** Optimized the Single Decision Tree by limiting its depth (`max_depth=3`), forcing it to focus only on big, overarching survival patterns. This boosted accuracy to 80.45%.
* **Linear Classification:** Tested a Logistic Regression model to draw clean mathematical boundaries, hitting a solid 81.01% accuracy.
* **Ensemble Learning (The Champion):** Scaled up to a Random Forest classifier. By adjusting its hyperparameters to `n_estimators=100` and `max_depth=10`, the model successfully utilized the "wisdom of the crowd" to handle deeper features without overfitting.

## 🏆 Final Model Leaderboard

| Model Name | Test Accuracy | Status |
| :--- | :---: | :--- |
| Original Decision Tree | 79.89% | Overfitted Baseline |
| Tuned Decision Tree (Depth 3) | 80.45% | Optimized Tree |
| Logistic Regression | 81.01% | Strong Linear Baseline |
| **Random Forest (Depth 10)** | **83.24%** | **🏆 Chosen Champion Model** |

## 🔑 Key Takeaway
Going from a single overfitted tree to a tuned Random Forest ensemble completely changed the game. Restricting tree parameters while letting 100 diverse trees vote on predictions allowed the model to comfortably clear the 83% accuracy threshold on unseen test data.

