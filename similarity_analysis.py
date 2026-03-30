import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

def compute_model_similarities(df_performance):
    """
    Computes similarity matrices for machine learning models based on 
    their performance across multiple datasets (Varun's Role).
    
    Args:
        df_performance (pd.DataFrame): DataFrame prepared by Daniel R.S.
                                       Rows = datasets, Columns = machine learning models.
                                       Values = performance metrics (e.g., accuracy).
    
    Returns:
        dict: A dictionary containing correlation, cosine similarity, and euclidean distance matrices.
    """
    # 1. Correlation (Pearson)
    # Measures how the performance of two models trend together across datasets
    corr_matrix = df_performance.corr(method='pearson')
    
    # 2. Cosine Similarity
    # Measures the angle between two performance vectors (alignment in behavior)
    # We transpose because cosine_similarity computes distance between rows, but our models are columns
    cos_sim_array = cosine_similarity(df_performance.T)
    cos_sim_matrix = pd.DataFrame(cos_sim_array, index=df_performance.columns, columns=df_performance.columns)
    
    # 3. Euclidean Distance
    # Measures the literal magnitude and distance difference between two performance vectors
    euc_dist_array = euclidean_distances(df_performance.T)
    euc_dist_matrix = pd.DataFrame(euc_dist_array, index=df_performance.columns, columns=df_performance.columns)
    
    return {
        'correlation': corr_matrix,
        'cosine_similarity': cos_sim_matrix,
        'euclidean_distance': euc_dist_matrix
    }

if __name__ == "__main__":
    # --- SIMULATED PIPELINE ---
    # This simulates receiving the cleaned data from Daniel Rodas Sanchez.
    # In the actual workflow, you will use pd.read_csv('cleaned_data.csv') here.
    
    print("Loading cleaned dataset (Simulated)...")
    np.random.seed(42)
    datasets = [f"Dataset_{i}" for i in range(1, 122)]
    models = ["RandomForest", "SVM", "NeuralNet", "DecisionTree", "KNN", "NaiveBayes", "AdaBoost", "LogReg"]
    
    # Generate dummy accuracy data between 0.5 and 1.0
    dummy_data = np.random.uniform(0.5, 1.0, size=(121, len(models)))
    df_cleaned = pd.DataFrame(dummy_data, index=datasets, columns=models)
    
    # --- VARUN's ROLE ---
    print("Computing similarity metrics between models...")
    similarities = compute_model_similarities(df_cleaned)
    
    # Save the output matrices for Daniel Si-Hyun Ryu (Clustering) and Zhiheng Shao (Plots)
    corr_file = 'model_correlation_matrix.csv'
    cos_file = 'model_cosine_similarity_matrix.csv'
    euc_file = 'model_euclidean_distance_matrix.csv'
    
    similarities['correlation'].to_csv(corr_file)
    similarities['cosine_similarity'].to_csv(cos_file)
    similarities['euclidean_distance'].to_csv(euc_file)
    
    print("\nSuccess! Handing off to the rest of the team:")
    print(f"- Saved {corr_file}")
    print(f"- Saved {cos_file}")
    print(f"- Saved {euc_file}")
    
    print("\nThese matrices can now be directly loaded for Clustering and Plotting.")
