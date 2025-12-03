# Dataset Classification using ML + Genetic Algorithm for Feature Selection

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2-green)
![DEAP](https://img.shields.io/badge/DEAP-1.3-orange)

---

## **Project Overview**
This project implements several Machine Learning models (KNN, NB, SVM, ANN, RF) for a classification problem. We also apply a Genetic Algorithm (GA) for feature selection and compare performance before and after feature reduction.

**Key objectives:**
- Train and evaluate four models: **KNN, SVM, Naive Bayes, ANN**  
- Apply GA to select the most relevant features  
- Compare performance **before and after GA**  
- Visualize **confusion matrices** and **metric comparison charts**


**Team Members**
- Member 1: Mario Ashraf

- Member 2: Mark Ashraf

- Member 3: Matthew Mokhles

- Member 4: Veronia Gamil

---

## **Dataset**
- Dataset 1: Breast Cancer Wisconsin (Diagnostic) Data Set, source: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
- Dataset 2: Heart Disease Dataset, https://archive.ics.uci.edu/dataset/45/heart+disease
- Dataset 3 source: UCI / Kaggle
- Dataset 4 source: UCI / Kaggle


**Preprocessing Applied:**
- Handling missing values
- Encoding categorical data
- Feature scaling/normalization
- Splitting into training/testing sets

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

## **Video Presentations**
- Mario: link
- Mark: link
- Matthew: link
- Veronia: links

---
