import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import random

# --- Global Configuration ---
RANDOM_STATE = 42
TEST_SIZE = 0.2

# --- 1. Dataset Acquisition, Description, and Preprocessing ---

def load_and_preprocess_data(file_path='D:/SEMESTER 7/Advanced AI/12th Project/Trial 2/data.csv'):
    """Loads, cleans, encodes, and scales the Breast Cancer Wisconsin dataset."""
    print("--- 1. Data Preprocessing ---")
    df = pd.read_csv(file_path)
    
    # Drop unnecessary columns
    df.drop(['id', 'Unnamed: 32'], axis=1, inplace=True, errors='ignore')
    
    # Separate features (X) and target (y)
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']
    
    # Encode target variable (M=1, B=0)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    print(f"Dataset shape: {X_scaled_df.shape}")
    return X_scaled_df, y_encoded


# --- Helper for Model Evaluation ---

def evaluate_model(X, y, model, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """Splits data, trains model, and reports performance metrics."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'Confusion Matrix': confusion_matrix(y_test, y_pred).tolist()
    }
    return metrics

# --- 2. Advanced Machine Learning Benchmarking and Selection ---

def get_models():
    """Defines and returns the dictionary of models."""
    return {
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Naïve Bayes': GaussianNB(),
        'SVM': SVC(kernel='linear', random_state=RANDOM_STATE),
        'ANN': MLPClassifier(hidden_layer_sizes=(50, 25),activation="relu", solver="adam",learning_rate_init=0.0005,max_iter=2000,early_stopping=True, random_state=RANDOM_STATE),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    }

def benchmark_models(X, y, models):
    """Evaluates multiple models and selects the best one based on F1-Score."""
    print("\n--- 2. Model Benchmarking (Original Dataset) ---")
    
    results = {}
    
    for name, model in models.items():
        metrics = evaluate_model(X, y, model)
        results[name] = metrics
        print(f"  {name}: Acc={metrics['Accuracy']:.4f}, F1={metrics['F1-Score']:.4f}")

    return results

# --- 3. Dimensionality Reduction using Genetic Algorithm (GA) ---

class SimpleGA:
    # Reduced pop_size and n_generations for faster execution in sandbox
    def __init__(self, X, y, model, pop_size=50, n_generations=20, cx_prob=0.8, mut_prob=0.01):
        self.X = X
        self.y = y
        self.model = model
        self.n_features = X.shape[1]
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.cx_prob = cx_prob
        self.mut_prob = mut_prob
        self.population = self._initialize_population()
        self.best_chromosome = None
        self.best_fitness = -1

    def _initialize_population(self):
        """Initializes a population of random binary chromosomes (feature masks)."""
        return [np.random.randint(0, 2, self.n_features) for _ in range(self.pop_size)]

    def _fitness(self, chromosome):
        """Evaluates the fitness of a chromosome (feature mask) using cross-validation F1-Score."""
        selected_indices = np.where(chromosome == 1)[0]
        if len(selected_indices) == 0:
            return 0.0 # Penalty for no features selected
        
        X_reduced = self.X.iloc[:, selected_indices]
        
        # Use cross-validation for a more robust fitness score
        try:
            scores = cross_val_score(self.model, X_reduced, self.y, cv=5, scoring='f1')
            return scores.mean()
        except ValueError:
            # Handle cases where the reduced dataset is invalid for the model
            return 0.0

    def _select(self, fitnesses):
        """Tournament selection (simplified)."""
        # Select parents based on fitness (higher is better)
        total_fitness = sum(fitnesses)
        if total_fitness == 0:
            # If all fitnesses are zero, select randomly
            probabilities = [1.0 / self.pop_size] * self.pop_size
        else:
            probabilities = [f / total_fitness for f in fitnesses]
            
        indices = np.random.choice(self.pop_size, size=self.pop_size, p=probabilities)
        return [self.population[i] for i in indices]

    def _crossover(self, parent1, parent2):
        """Single-point crossover."""
        if random.random() < self.cx_prob:
            point = random.randint(1, self.n_features - 1)
            child1 = np.concatenate((parent1[:point], parent2[point:]))
            child2 = np.concatenate((parent2[:point], parent1[point:]))
            return child1, child2
        return parent1.copy(), parent2.copy()

    def _mutate(self, chromosome):
        """Bit-flip mutation."""
        for i in range(self.n_features):
            if random.random() < self.mut_prob:
                chromosome[i] = 1 - chromosome[i]
        return chromosome

    def run(self):
        """Runs the Genetic Algorithm."""
        print("\n--- 3. Genetic Algorithm Feature Selection ---")
        
        for gen in range(self.n_generations):
            fitnesses = [self._fitness(c) for c in self.population]
            
            # Update best chromosome
            current_best_fitness = max(fitnesses)
            if current_best_fitness > self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_chromosome = self.population[fitnesses.index(current_best_fitness)]
            
            print(f"  Generation {gen+1}/{self.n_generations}: Best Fitness = {self.best_fitness:.4f}, Features = {np.sum(self.best_chromosome)}")

            # Selection
            offspring = self._select(fitnesses)
            
            # Crossover and Mutation
            new_population = []
            for i in range(0, self.pop_size, 2):
                p1 = offspring[i]
                p2 = offspring[i+1] if i+1 < self.pop_size else offspring[i] # Handle odd population size
                
                c1, c2 = self._crossover(p1, p2)
                
                new_population.append(self._mutate(c1))
                if i+1 < self.pop_size:
                    new_population.append(self._mutate(c2))
            
            self.population = new_population[:self.pop_size] # Ensure population size is maintained

        print("\nGA Complete.")
        return self.best_chromosome

# --- 4. Independent GA Feature Selection and Evaluation ---

def run_independent_ga_selection(X, y, models, original_benchmarks):
    """
    Runs GA feature selection independently for each model and evaluates performance
    on the resulting reduced feature set.
    """
    print("\n--- 4. Independent GA Feature Selection and Evaluation ---")
    
    full_results = {}
    
    for name, model in models.items():
        print(f"\n--- Running GA for {name} ---")
        
        # 4.1. Run GA Feature Selection using the model's performance as fitness
        ga = SimpleGA(X, y, model)
        best_feature_mask = ga.run()
        
        # 4.2. Apply the feature mask
        selected_indices = np.where(best_feature_mask == 1)[0]
        reduced_dimension = len(selected_indices)
        
        if reduced_dimension == 0:
            print(f"  Warning: GA for {name} selected no features. Skipping reduced evaluation.")
            reduced_metrics = {'Accuracy': 0.0, 'Precision': 0.0, 'Recall': 0.0, 'F1-Score': 0.0, 'Confusion Matrix': 'N/A'}
        else:
            X_reduced = X.iloc[:, selected_indices]
            
            # 4.3. Retrain and evaluate the model on the reduced dataset
            reduced_metrics = evaluate_model(X_reduced, y, model)
            
            print(f"  {name} (GA-Reduced): Features={reduced_dimension}, Acc={reduced_metrics['Accuracy']:.4f}, F1={reduced_metrics['F1-Score']:.4f}")
        
        # Store results
        full_results[name] = {
            'original': original_benchmarks[name],
            'reduced': reduced_metrics,
            'mask': best_feature_mask,
            'dimension': reduced_dimension
        }
        
    return full_results

# --- 5. Final Comparative Analysis and Insights ---

def format_confusion_matrix(cm_list):
    """Formats the confusion matrix list into a readable string."""
    if cm_list == 'N/A':
        return 'N/A'
    
    # Assuming a 2x2 matrix: [[TN, FP], [FN, TP]]
    # Class 0 (Benign), Class 1 (Malignant)
    cm = np.array(cm_list)
    
    # Create a simple Markdown table representation
    header = "| | Predicted 0 (B) | Predicted 1 (M) |"
    separator = "|---|---|---|"
    row0 = f"| **Actual 0 (B)** | {cm[0, 0]} (TN) | {cm[0, 1]} (FP) |"
    row1 = f"| **Actual 1 (M)** | {cm[1, 0]} (FN) | {cm[1, 1]} (TP) |"
    
    return "\n".join([header, separator, row0, row1])

def final_comparative_analysis(full_results, X_scaled):
    """Presents a comprehensive table comparing all models' performance."""
    print("\n--- 5. Final Comprehensive Comparative Analysis ---")
    
    data = []
    for name, res in full_results.items():
        original = res['original']
        reduced = res['reduced']
        dim = res['dimension']
        
        data.append({
            'Model': name,
            'Features': 'Original (30)',
            'Accuracy': original['Accuracy'],
            'F1-Score': original['F1-Score'],
            'Precision': original['Precision'],
            'Recall': original['Recall'],
        })
        data.append({
            'Model': name,
            'Features': f'GA-Reduced ({dim})',
            'Accuracy': reduced['Accuracy'],
            'F1-Score': reduced['F1-Score'],
            'Precision': reduced['Precision'],
            'Recall': reduced['Recall'],
        })
        
    comparison_df = pd.DataFrame(data)
    
    # Format the table for better readability
    comparison_df_formatted = comparison_df.style.format({
        'Accuracy': "{:.4f}",
        'F1-Score': "{:.4f}",
        'Precision': "{:.4f}",
        'Recall': "{:.4f}",
    }).hide(axis='index').to_html() # Using to_html for better formatting in a markdown environment
    
    print("\nPerformance Comparison (Original vs. GA-Reduced Feature Sets):")
    print(comparison_df.to_markdown(index=False, floatfmt=".4f"))
    
    print("\nDetailed Performance and Confusion Matrices:")
    
    for name, res in full_results.items():
        original = res['original']
        reduced = res['reduced']
        dim = res['dimension']
        
        print(f"\n### Model: {name}")
        
        # Original Dataset
        print(f"\n#### Original Dataset (30 Features)")
        print(f"  - Accuracy: {original['Accuracy']:.4f}")
        print(f"  - Precision: {original['Precision']:.4f}")
        print(f"  - Recall: {original['Recall']:.4f}")
        print(f"  - F1-Score: {original['F1-Score']:.4f}")
        print("\nConfusion Matrix:")
        print(format_confusion_matrix(original['Confusion Matrix']))
        
        # GA-Reduced Dataset
        print(f"\n#### GA-Reduced Dataset ({dim} Features)")
        print(f"  - Accuracy: {reduced['Accuracy']:.4f}")
        print(f"  - Precision: {reduced['Precision']:.4f}")
        print(f"  - Recall: {reduced['Recall']:.4f}")
        print(f"  - F1-Score: {reduced['F1-Score']:.4f}")
        print("\nConfusion Matrix:")
        print(format_confusion_matrix(reduced['Confusion Matrix']))
    
    print("\nFeature Subsets (Masks):")
    for name, res in full_results.items():
        # Ensure the mask is a numpy array for correct indexing
        mask = np.array(res['mask'])
        feature_names = X_scaled.columns[np.where(mask == 1)[0]].tolist()
        print(f"  {name} ({res['dimension']} features): {', '.join(feature_names)}")


# --- Main Execution ---
if __name__ == "__main__":
    # 1. Load and Preprocess Data
    X_scaled, y_encoded = load_and_preprocess_data()
    
    # 2. Get Models
    models = get_models()
    
    # 3. Benchmark Models on Original Data
    original_benchmarks = benchmark_models(X_scaled, y_encoded, models)
    
    # 4. Run Independent GA Feature Selection and Evaluation
    full_results = run_independent_ga_selection(X_scaled, y_encoded, models, original_benchmarks)
    
    # 5. Final Comparative Analysis
    final_comparative_analysis(full_results, X_scaled)
    
