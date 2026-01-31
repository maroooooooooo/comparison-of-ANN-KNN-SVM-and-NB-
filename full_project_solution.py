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

def benchmark_models(X, y):
    """Evaluates multiple models and selects the best one based on F1-Score."""
    print("\n--- 2. Model Benchmarking (Original Dataset) ---")
    
    # Define models with reasonable starting hyperparameters
    models = {
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Naïve Bayes': GaussianNB(),
        'SVM': SVC(kernel='linear', random_state=RANDOM_STATE),
        'ANN': MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=RANDOM_STATE),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    }
    
    results = {}
    
    for name, model in models.items():
        metrics = evaluate_model(X, y, model)
        results[name] = metrics
        print(f"  {name}: Acc={metrics['Accuracy']:.4f}, F1={metrics['F1-Score']:.4f}, CM={metrics['Confusion Matrix']}")

    # Select the best model based on F1-Score
    best_model_name = max(results, key=lambda name: results[name]['F1-Score'])
    best_model_metrics = results[best_model_name]
    best_model = models[best_model_name]
    
    print(f"\n--- Best Model Selected: {best_model_name} (F1-Score: {best_model_metrics['F1-Score']:.4f}) ---")
    
    return best_model_name, best_model, best_model_metrics

# --- 3. Dimensionality Reduction using Genetic Algorithm (GA) ---

class SimpleGA:
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

# --- 4. ML Implementation on GA-Reduced Dataset ---

def run_ga_reduced_ml(X, y, best_model, feature_mask):
    """Runs ML on the GA-reduced dataset."""
    print("\n--- 4. ML Implementation (GA-Reduced Features) ---")
    
    # Apply the feature mask
    selected_indices = np.where(feature_mask == 1)[0]
    X_reduced = X.iloc[:, selected_indices]
    
    # Retrain and evaluate the best model on the reduced dataset
    reduced_metrics = evaluate_model(X_reduced, y, best_model)
    
    print(f"Features used: {X_reduced.shape[1]}")
    print(f"Metrics: Acc={reduced_metrics['Accuracy']:.4f}, F1={reduced_metrics['F1-Score']:.4f}, CM={reduced_metrics['Confusion Matrix']}")
    return reduced_metrics, X_reduced.shape[1]

# --- 5. Comparative Analysis and Insights ---

def comparative_analysis(best_model_name, original_metrics, reduced_metrics, reduced_dimension):
    """Compares the performance metrics."""
    print("\n--- 5. Comparative Analysis ---")
    
    comparison_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Original (30 Features)': [original_metrics[m] for m in ['Accuracy', 'Precision', 'Recall', 'F1-Score']],
        f'GA-Reduced ({reduced_dimension} Features)': [reduced_metrics[m] for m in ['Accuracy', 'Precision', 'Recall', 'F1-Score']]
    })
    
    print(f"\nPerformance Comparison for {best_model_name}:")
    print(comparison_df.to_markdown(index=False, floatfmt=".4f"))
    
    print("\nConfusion Matrix Comparison:")
    print(f"Original CM: {original_metrics['Confusion Matrix']}")
    print(f"GA-Reduced CM: {reduced_metrics['Confusion Matrix']}")
    
    


# --- Main Execution ---
if __name__ == "__main__":
    # 1. Load and Preprocess Data
    X_scaled, y_encoded = load_and_preprocess_data()
    
    # 2. Benchmark Models and Select Best
    best_name, best_model, original_metrics = benchmark_models(X_scaled, y_encoded)
    
    # 3. Run GA Feature Selection
    ga = SimpleGA(X_scaled, y_encoded, best_model)
    best_feature_mask = ga.run()
    
    # 4. Run ML on GA-Reduced Dataset
    reduced_metrics, reduced_dimension = run_ga_reduced_ml(X_scaled, y_encoded, best_model, best_feature_mask)
    
    # 5. Comparative Analysis
    comparative_analysis(best_name, original_metrics, reduced_metrics, reduced_dimension)
    
