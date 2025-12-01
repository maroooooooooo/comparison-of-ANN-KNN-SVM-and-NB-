# Supermarket Customer Gender Classification with GA Feature Selection

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2-green)
![DEAP](https://img.shields.io/badge/DEAP-1.3-orange)

---

## **Project Overview**
This project predicts **customer gender** in a supermarket dataset using machine learning. It applies **Genetic Algorithm (GA) feature selection** to improve model performance.  

**Key objectives:**
- Train and evaluate four models: **KNN, SVM, Naive Bayes, ANN**  
- Apply GA to select the most relevant features  
- Compare performance **before and after GA**  
- Visualize **confusion matrices** and **metric comparison charts**  

---

## **Dataset**
- File: `SuperMarket Analysis.csv`  
- Target variable: `Gender` (binary)  
- Features: Categorical and numerical customer data  
- Missing values are imputed with the **mode**  
- Categorical columns are encoded using **LabelEncoder**  

---

## **Dependencies**
```bash
pip install pandas numpy scikit-learn seaborn matplotlib deap
```

---

## **Usage**
1. Place `SuperMarket Analysis.csv` in the project folder.  
2. Run the script:
```bash
python supermarket_ga_analysis.py
```
3. Outputs:
- Metrics for all four models (before & after GA)  
- Individual confusion matrices for each model  
- Bar charts comparing metrics before and after GA  

---

## **Code Highlights**
- **Preprocessing:** drop duplicates, handle missing values, encode categories, normalize features  
- **Evaluation:** Accuracy, Precision, Recall, F1, Confusion Matrix  
- **GA Feature Selection:** SVM-based fitness function, 20 population size, 10 generations  
- **Visualization:** Confusion matrices and grouped bar charts for metrics comparison  

---

## **Customization**
- Change target variable: update `target` in the script  
- Tune GA: adjust `POP_SIZE`, `N_GEN`, `CX_PB`, `MUT_PB` for better results  
- Models and hyperparameters can be modified in `evaluate_models()`  

---
