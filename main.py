import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------
# 1. Define the Models and Data (Decision Matrix)
# ---------------------------------------------------
# Data sourced from official SBERT benchmarks
data = {
    'Model': [
        'all-mpnet-base-v2', 
        'all-MiniLM-L6-v2', 
        'all-distilroberta-v1', 
        'paraphrase-albert-small-v2', 
        'multi-qa-mpnet-base-dot-v1'
    ],
    # Criterion 1: Accuracy (Higher is better)
    'Cosine_Similarity': [0.69, 0.68, 0.68, 0.64, 0.66], 
    # Criterion 2: Speed (Sentences/sec) (Higher is better)
    'Inference_Speed': [2800, 14200, 4000, 5000, 2800],
    # Criterion 3: Size (MB) (Lower is better)
    'Model_Size_MB': [420, 80, 290, 43, 420]
}

df = pd.DataFrame(data)

# ---------------------------------------------------
# 2. TOPSIS Algorithm Implementation
# ---------------------------------------------------

def apply_topsis(df, weights, impacts):
    """
    df: pandas DataFrame
    weights: list of weights for criteria (must sum to 1)
    impacts: list of '+' for beneficial criteria, '-' for non-beneficial
    """
    # Working on a copy of numeric data
    dft = df.iloc[:, 1:].copy()
    
    # Step 2.1: Vector Normalization
    # Formula: x_ij / sqrt(sum(x_kj^2))
    norm_dft = dft / np.sqrt((dft**2).sum())
    
    # Step 2.2: Weight Assignment
    weighted_dft = norm_dft * weights
    
    # Step 2.3: Find Ideal Best (V+) and Ideal Worst (V-)
    ideal_best = []
    ideal_worst = []
    
    for i, col in enumerate(weighted_dft.columns):
        if impacts[i] == '+':
            ideal_best.append(weighted_dft[col].max())
            ideal_worst.append(weighted_dft[col].min())
        else: # impact is '-' (lower is better)
            ideal_best.append(weighted_dft[col].min())
            ideal_worst.append(weighted_dft[col].max())
            
    # Step 2.4: Euclidean Distance Calculation
    # Distance from Best (S+)
    s_plus = np.sqrt(((weighted_dft - ideal_best) ** 2).sum(axis=1))
    
    # Distance from Worst (S-)
    s_minus = np.sqrt(((weighted_dft - ideal_worst) ** 2).sum(axis=1))
    
    # Step 2.5: Calculate Performance Score (P)
    # Formula: S- / (S+ + S-)
    topsis_score = s_minus / (s_plus + s_minus)
    
    return topsis_score

# Configuration for Text Sentence Similarity
# Weights: [Accuracy, Speed, Size] -> Accuracy is usually most important
weights = [0.5, 0.3, 0.2] 

# Impacts: [+, +, -] -> Accuracy (+), Speed (+), Size (-)
impacts = ['+', '+', '-']

# Calculate Scores
df['TOPSIS_Score'] = apply_topsis(df, weights, impacts)

# Rank the models (Higher score is better)
df['Rank'] = df['TOPSIS_Score'].rank(ascending=False).astype(int)

# Sort by Rank
df_sorted = df.sort_values(by='Rank')

print("Final Rankings:\n")
print(df_sorted)

# ---------------------------------------------------
# 3. Generate Outputs for GitHub
# ---------------------------------------------------

# Save to CSV
df_sorted.to_csv('topsis_results.csv', index=False)
print("\n[Success] Results saved to 'topsis_results.csv'")

# Generate Bar Chart
plt.figure(figsize=(10, 6))
sns.barplot(x='TOPSIS_Score', y='Model', data=df_sorted, palette='viridis')
plt.title('TOPSIS Ranking of Pre-Trained Models for Text Sentence Similarity')
plt.xlabel('TOPSIS Score (Higher is Better)')
plt.ylabel('Model Name')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save plot
plt.savefig('topsis_ranking_graph.png')
print("[Success] Graph saved to 'topsis_ranking_graph.png'")
plt.show()