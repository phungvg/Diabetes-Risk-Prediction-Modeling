# 🩺 Diabetes Risk and Medical Cost Modeling

IDE: R Studio and R Shiny

Predict diabetes (0/1) with Logistic, Radial Supper Vector Machine, Decision Tree, Random Forest, Naive Bayes and tune a clinical decision threshold (precision–recall trade-off).

Estimate medical charges with linear regression, compare feature sets, and report RMSE/R².

Includes ROC/AUC, confusion matrix, and reproducible train/test splits.

datasets: diabetes.csv

---

## ⚙️ Setup

### 🔁 1. Clone the Repository

Use Git to download the project from GitHub:

```bash
git clone https://github.com/phungvg/diabetes-risk-predction-modeling.git

```

### ☁️ 2. R Packages
Run locally in app.R

```bash
install.packages(c(
  "shiny","caret","randomForest","MASS","e1071","kernlab", "rpart","klaR","ggplot2","dplyr","tidyr"
  # optional: "shinythemes"
))
```


### ▶️ Run the App

- [x] **Diabetes(classification)**  
  - Filter out your current situation:
  - 
       Demographics: AgeCategory(1=18-24, ..., 13=80+, Sex, BMI,High Blood Pressure(yes/no), High Cholesterol(yes/no)
    
       Lifestyle: Smoker(Lifetime >100cigs, yes/no), Physical Acitivity (Past 30 days yes/no), Daily Fruit/Veggie Consumption (yes/no), Heavy Alcohol Consumption(yes/no)

       History & Health: General Health (1: Excellent, 2: Very good, 3: Good, 4: Fair, 5: Poor), Days Poor Mental Health(0-30), Days Poor Physical Health (0-30), Cholesterol Check(5y), History of Stroke (yes/no), Heart Disease/Attack (yes/no), Difficulty Walking(yes/no)


  



---
### Visualization from the app
| Diabetes tab | Medical Cost tab |
Diabetes tab — choose classifier, tune threshold, view ROC/AUC & metrics
<img src= "https://github.com/user-attachments/assets/ffa87c0b-2ef5-4494-929b-c72d04a83e2e" alt="Diabetes tab: model, threshold, ROC/AUC" width="100%">


