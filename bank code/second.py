# 1. Import libraries
import pandas as pd
import numpy as np
import random
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

# 2. Load dataset
df = pd.read_csv("bank.csv", sep=';')

# 3. Preprocessing
X = df.drop("y", axis=1)
y = df["y"]

y = LabelEncoder().fit_transform(y)

categorical_cols = X.select_dtypes(include=['object']).columns
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns

# Scale numeric features
scaler = StandardScaler()
X_numeric = scaler.fit_transform(X[numeric_cols])

# One-hot encode categorical features
ohe = OneHotEncoder(drop='first', sparse_output=False)
X_categorical = ohe.fit_transform(X[categorical_cols])

# Combine
X = np.hstack([X_numeric, X_categorical])

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train 5 Classifiers
models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "NaiveBayes": GaussianNB(),
    "SVM": SVC(kernel='rbf', C=1, gamma='scale'),
    "ANN": MLPClassifier(hidden_layer_sizes=(20,10), activation='relu', max_iter=2000,solver='adam'),
    "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    results[name] = {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "Confusion Matrix": cm,
        "Model Object": model
    }

    print(f"\n==============================")
    print(f"{name} RESULTS")
    print("==============================")
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1 Score:", f1)
    print("Confusion Matrix:\n", cm)

# 5. Select Best Model Based on F1-score
best_model_name = max(results, key=lambda x: results[x]["F1 Score"])
best_model_object = results[best_model_name]["Model Object"]

print("\n\n########################################")
print(" BEST MODEL BASED ON F1-SCORE:", best_model_name)
print("########################################")
print("Best Model Performance:", results[best_model_name])

# 6. Genetic Algorithm Feature Selection
num_features = X.shape[1]

POP_SIZE = 30
GENERATIONS = 40
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.1

def fitness_function(chromosome):
    selected_idx = [i for i in range(len(chromosome)) if chromosome[i] == 1]

    if len(selected_idx) == 0:
        return 0

    X_train_sel = X_train[:, selected_idx]
    X_test_sel = X_test[:, selected_idx]

    model = best_model_object.__class__(**best_model_object.get_params())  # fresh copy
    model.fit(X_train_sel, y_train)
    y_pred = model.predict(X_test_sel)

    return f1_score(y_test, y_pred)

def init_population():
    return [ [random.randint(0,1) for _ in range(num_features)] for _ in range(POP_SIZE) ]

def crossover(parent1, parent2):
    if random.random() > CROSSOVER_RATE:
        return parent1, parent2
    point = random.randint(1, num_features-1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

def mutate(chromosome):
    for i in range(num_features):
        if random.random() < MUTATION_RATE:
            chromosome[i] = 1 - chromosome[i]
    return chromosome

# GA loop
population = init_population()

for gen in range(GENERATIONS):
    fitness_scores = [fitness_function(ind) for ind in population]

    new_population = []

    elite_idx = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[:2]
    new_population.append(population[elite_idx[0]])
    new_population.append(population[elite_idx[1]])

    while len(new_population) < POP_SIZE:
        parents = random.sample(population, 2)
        child1, child2 = crossover(parents[0], parents[1])
        child1 = mutate(child1)
        child2 = mutate(child2)
        new_population.append(child1)
        new_population.append(child2)

    population = new_population[:POP_SIZE]
    print(f"Generation {gen+1}/{GENERATIONS} | Best F1-Score = {max(fitness_scores):.4f}")

# Best Chromosome
fitness_scores = [fitness_function(ind) for ind in population]
best_chrom = population[fitness_scores.index(max(fitness_scores))]

selected_features = [i for i in range(len(best_chrom)) if best_chrom[i] == 1]

print("\n==============================")
print(" BEST FEATURE SUBSET SELECTED BY GA")
print("==============================")
print("Number of selected features:", len(selected_features))
print("Selected feature indices:", selected_features)

# 7. Train the BEST MODEL on GA-Reduced Dataset

X_train_reduced = X_train[:, selected_features]
X_test_reduced = X_test[:, selected_features]

# Create a fresh copy of the winning model
final_model = best_model_object.__class__(**best_model_object.get_params())

# Train
final_model.fit(X_train_reduced, y_train)
y_pred_ga = final_model.predict(X_test_reduced)

# Metrics
acc = accuracy_score(y_test, y_pred_ga)
prec = precision_score(y_test, y_pred_ga)
rec = recall_score(y_test, y_pred_ga)
f1 = f1_score(y_test, y_pred_ga)
cm = confusion_matrix(y_test, y_pred_ga)

print("\n==============================")
print(f" {best_model_name} ON GA-REDUCED FEATURES")
print("==============================")
print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("F1 Score:", f1)
print("Confusion Matrix:\n", cm)


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Example set of supported models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# 1) AUTO-SELECT SAME MODEL FROM REQUIREMENT #2
def auto_select_model(model_name):
    model_name = model_name.lower()

    if "logistic" in model_name:
        return LogisticRegression(max_iter=2000)
    elif "svm" in model_name:
        return SVC()
    elif "knn" in model_name:
        return KNeighborsClassifier()
    elif "forest" in model_name or "rf" in model_name or "random" in model_name:
        return RandomForestClassifier()
    else:
        raise ValueError("ERROR: Unknown model name. Add it in auto_select_model().")

# 2) FUNCTION TO TRAIN + GET PERFORMANCE METRICS
def evaluate(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="macro"),
        "Recall": recall_score(y_test, y_pred, average="macro"),
        "F1 Score": f1_score(y_test, y_pred, average="macro"),
        "Confusion Matrix": confusion_matrix(y_test, y_pred)
    }
    return results

# 3) LOAD ORIGINAL AND GA-REDUCED DATASETS
# IMPORTANT: Replace these with your actual files
X_original = pd.read_csv("original_features.csv").values
y = pd.read_csv("labels.csv").values.ravel()

X_ga_reduced = pd.read_csv("ga_reduced_features.csv").values

# SAME split for fair comparison
X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(
    X_original, y, test_size=0.2, random_state=42)

X_train_ga, X_test_ga, y_train_ga, y_test_ga = train_test_split(
    X_ga_reduced, y, test_size=0.2, random_state=42)

# 4) RUN MODEL ON ORIGINAL + GA DATASETS
selected_model_name = "Random Forest"   # This can be filled automatically in Requirement 2
model = auto_select_model(selected_model_name)

# Performance before GA
performance_original = evaluate(model, X_train_o, X_test_o, y_train_o, y_test_o)

# Performance after GA
performance_ga = evaluate(model, X_train_ga, X_test_ga, y_train_ga, y_test_ga)

# 5) PRINT RESULTS CLEARLY
print("\n================= ORIGINAL DATASET PERFORMANCE =================")
for metric, value in performance_original.items():
    print(f"{metric}:\n{value}\n")

print("\n================= GA-REDUCED DATASET PERFORMANCE =================")
for metric, value in performance_ga.items():
    print(f"{metric}:\n{value}\n")

# 6) COMPARISON TABLE
comparison = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
    "Original Dataset": [
        performance_original["Accuracy"],
        performance_original["Precision"],
        performance_original["Recall"],
        performance_original["F1 Score"],
    ],
    "GA-Reduced Dataset": [
        performance_ga["Accuracy"],
        performance_ga["Precision"],
        performance_ga["Recall"],
        performance_ga["F1 Score"],
    ]
})

print("\n================= COMPARISON TABLE =================")
print(comparison)
