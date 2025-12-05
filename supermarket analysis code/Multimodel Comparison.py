# ============================================
# 1. Import libraries
# ============================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import seaborn as sns
import matplotlib.pyplot as plt

# GA libraries
from deap import base, creator, tools, algorithms
import random

# ============================================
# 2. Load dataset
# ============================================
df = pd.read_csv("SuperMarket Analysis.csv")

# ============================================
# 3. Basic cleaning
# ============================================
df = df.drop_duplicates()
df = df.fillna(df.mode().iloc[0])

# ============================================
# 4. Encode categorical columns
# ============================================
label_enc = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = label_enc.fit_transform(df[col])

# ============================================
# 5. Split features and target
# ============================================
target = "Gender"   # <<=== Change if needed
X = df.drop(columns=[target])
y = df[target]

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================
# 6. Function to train & evaluate models
# ============================================
def evaluate_models(X_tr, X_te, y_tr, y_te):
    results = {}

    models = {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(kernel='rbf'),
        "Naive Bayes": GaussianNB(),
        "ANN": MLPClassifier(hidden_layer_sizes=(32,16), max_iter=700)
    }

    for name, model in models.items():
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        results[name] = {
            "Accuracy": accuracy_score(y_te, y_pred),
            "Precision": precision_score(y_te, y_pred, average='weighted', zero_division=0),
            "Recall": recall_score(y_te, y_pred, average='weighted', zero_division=0),
            "F1-Score": f1_score(y_te, y_pred, average='weighted', zero_division=0),
            "Confusion Matrix": confusion_matrix(y_te, y_pred)
        }
    return results

# ============================================
# 7. Confusion matrix plot
# ============================================
def plot_cm(cm, title):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# ============================================
# 8. Evaluate BEFORE GA
# ============================================
print("=== Performance BEFORE GA Feature Selection ===")
results_before = evaluate_models(X_train, X_test, y_train, y_test)

for model_name, metrics in results_before.items():
    print(f"\n{model_name}")
    for metric_name, value in metrics.items():
        if metric_name != "Confusion Matrix":
            print(f"{metric_name}: {value:.4f}")
    plot_cm(metrics["Confusion Matrix"], f"{model_name} Confusion Matrix (Before GA)")

# ============================================
# 9. GA Feature Selection (with CROSS-VALIDATION)
# ============================================
POP_SIZE = 20
N_GEN = 10
CX_PB = 0.5
MUT_PB = 0.2

n_features = X_train.shape[1]

# GA Fitness (maximize cross-val accuracy)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Each gene is 0 or 1 (include / exclude feature)
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=n_features)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# --- CROSS VALIDATION FIX ---
def eval_individual(individual):
    selected_features = [i for i, bit in enumerate(individual) if bit == 1]
    if len(selected_features) == 0:
        return 0.0,

    X_sel = X_train[:, selected_features]
    model = SVC(kernel='rbf')

    scores = cross_val_score(model, X_sel, y_train, cv=5)

    return scores.mean(),

toolbox.register("evaluate", eval_individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# Run GA
population = toolbox.population(n=POP_SIZE)
algorithms.eaSimple(population, toolbox, cxpb=CX_PB, mutpb=MUT_PB, ngen=N_GEN, verbose=False)

# Best solution
best_ind = tools.selBest(population, 1)[0]
selected_features = [i for i, bit in enumerate(best_ind) if bit == 1]

print("\nSelected Features by GA:", selected_features)

# Reduce dataset
X_train_ga = X_train[:, selected_features]
X_test_ga  = X_test[:, selected_features]

# ============================================
# 10. Evaluate AFTER GA
# ============================================
print("\n=== Performance AFTER GA Feature Selection ===")
results_after = evaluate_models(X_train_ga, X_test_ga, y_train, y_test)

for model_name, metrics in results_after.items():
    print(f"\n{model_name}")
    for metric_name, value in metrics.items():
        if metric_name != "Confusion Matrix":
            print(f"{metric_name}: {value:.4f}")
    plot_cm(metrics["Confusion Matrix"], f"{model_name} Confusion Matrix (After GA)")
# ============================================
# 11. Comparison Table BEFORE vs AFTER GA
# ============================================

comparison_data = []

for model_name in results_before.keys():
    comparison_data.append([
        model_name,
        results_before[model_name]["Accuracy"],
        results_after[model_name]["Accuracy"],
        results_before[model_name]["Precision"],
        results_after[model_name]["Precision"],
        results_before[model_name]["Recall"],
        results_after[model_name]["Recall"],
        results_before[model_name]["F1-Score"],
        results_after[model_name]["F1-Score"]
    ])

comparison_df = pd.DataFrame(
    comparison_data,
    columns=[
        "Model",
        "Accuracy (Before GA)", "Accuracy (After GA)",
        "Precision (Before GA)", "Precision (After GA)",
        "Recall (Before GA)", "Recall (After GA)",
        "F1 (Before GA)", "F1 (After GA)"
    ]
)

print("\n==================== COMPARISON TABLE ====================")
print(comparison_df)

# ============================================
# 10. Column chart comparing BEFORE vs AFTER GA
# ============================================

# Metrics to plot
metrics = ["Accuracy", "Precision", "Recall", "F1"]
models = comparison_df["Model"].tolist()

# Data for plotting
before_values = comparison_df[["Accuracy (Before GA)", "Precision (Before GA)", "Recall (Before GA)", "F1 (Before GA)"]].values
after_values = comparison_df[["Accuracy (After GA)", "Precision (After GA)", "Recall (After GA)", "F1 (After GA)"]].values

# Number of models and metrics
n_models = len(models)
n_metrics = len(metrics)
bar_width = 0.35
x = np.arange(n_models)

# Plot each metric
fig, axes = plt.subplots(2, 2, figsize=(14,10))
axes = axes.flatten()

colors_before = "#1f77b4"  # blue
colors_after = "#ff7f0e"   # orange

for i, metric in enumerate(metrics):
    ax = axes[i]
    ax.bar(x - bar_width/2, before_values[:, i], width=bar_width, color=colors_before, label="Before GA")
    ax.bar(x + bar_width/2, after_values[:, i], width=bar_width, color=colors_after, label="After GA")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} Comparison Before vs After GA")
    ax.legend()
    ax.set_ylim(0,1.05)  # metrics are between 0 and 1
    for j in range(n_models):
        ax.text(j - bar_width/2, before_values[j, i] + 0.02, f"{before_values[j, i]:.2f}", ha='center')
        ax.text(j + bar_width/2, after_values[j, i] + 0.02, f"{after_values[j, i]:.2f}", ha='center')

plt.tight_layout()
plt.show()

