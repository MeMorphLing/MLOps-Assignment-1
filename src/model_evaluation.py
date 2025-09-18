import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def compare_models():
    """Compare all models and create comparison plots"""
    # Use local MLflow tracking (no server needed)
    # mlflow.set_tracking_uri("http://localhost:5000")  # Comment out this line
    
    # Get experiment
    try:
        experiment = mlflow.get_experiment_by_name("Wine_Classification_Experiment")
        if experiment is None:
            print("Experiment not found!")
            print("Available experiments:")
            experiments = mlflow.search_experiments()
            for exp in experiments:
                print(f"- {exp.name}")
            return
    except Exception as e:
        print(f"Error getting experiment: {e}")
        print("Available experiments:")
        try:
            experiments = mlflow.search_experiments()
            for exp in experiments:
                print(f"- {exp.name}")
        except:
            print("No experiments found. Make sure you've run the training script first.")
        return
    
    # Get all runs
    try:
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        
        if runs.empty:
            print("No runs found in the experiment!")
            return
        
        print(f"Found {len(runs)} runs in the experiment")
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Get model names and metrics
        model_names = runs['tags.mlflow.runName'].fillna('Unknown')
        accuracy = runs['metrics.accuracy'].fillna(0)
        precision = runs['metrics.precision'].fillna(0)
        recall = runs['metrics.recall'].fillna(0)
        f1_score = runs['metrics.f1_score'].fillna(0)
        
        # Accuracy comparison
        axes[0,0].bar(model_names, accuracy, color=['skyblue', 'lightgreen', 'salmon'])
        axes[0,0].set_title('Model Accuracy Comparison')
        axes[0,0].set_ylabel('Accuracy')
        axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,0].set_ylim(0, 1.1)
        
        # Add value labels on bars
        for i, v in enumerate(accuracy):
            axes[0,0].text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')
        
        # Precision comparison
        axes[0,1].bar(model_names, precision, color=['skyblue', 'lightgreen', 'salmon'])
        axes[0,1].set_title('Model Precision Comparison')
        axes[0,1].set_ylabel('Precision')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].set_ylim(0, 1.1)
        
        # Add value labels on bars
        for i, v in enumerate(precision):
            axes[0,1].text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')
        
        # Recall comparison
        axes[1,0].bar(model_names, recall, color=['skyblue', 'lightgreen', 'salmon'])
        axes[1,0].set_title('Model Recall Comparison')
        axes[1,0].set_ylabel('Recall')
        axes[1,0].tick_params(axis='x', rotation=45)
        axes[1,0].set_ylim(0, 1.1)
        
        # Add value labels on bars
        for i, v in enumerate(recall):
            axes[1,0].text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')
        
        # F1-Score comparison
        axes[1,1].bar(model_names, f1_score, color=['skyblue', 'lightgreen', 'salmon'])
        axes[1,1].set_title('Model F1-Score Comparison')
        axes[1,1].set_ylabel('F1-Score')
        axes[1,1].tick_params(axis='x', rotation=45)
        axes[1,1].set_ylim(0, 1.1)
        
        # Add value labels on bars
        for i, v in enumerate(f1_score):
            axes[1,1].text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save the comparison plot
        os.makedirs('results', exist_ok=True)
        plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
        print("✅ Model comparison plot saved to: results/model_comparison.png")
        plt.show()
        
        # Print detailed comparison
        print("\n" + "="*60)
        print("DETAILED MODEL COMPARISON")
        print("="*60)
        
        for idx, run in runs.iterrows():
            model_name = run['tags.mlflow.runName'] if pd.notna(run['tags.mlflow.runName']) else 'Unknown'
            print(f"\n📊 Model: {model_name}")
            print(f"   Accuracy:  {run['metrics.accuracy']:.4f}")
            print(f"   Precision: {run['metrics.precision']:.4f}")
            print(f"   Recall:    {run['metrics.recall']:.4f}")
            print(f"   F1-Score:  {run['metrics.f1_score']:.4f}")
            print("-" * 50)
        
        # Find and highlight best model
        best_idx = accuracy.idxmax()
        best_model = runs.loc[best_idx]
        best_name = best_model['tags.mlflow.runName'] if pd.notna(best_model['tags.mlflow.runName']) else 'Unknown'
        
        print(f"\n🏆 BEST MODEL: {best_name}")
        print(f"   Best Accuracy: {best_model['metrics.accuracy']:.4f}")
        print("="*60)
        
        return runs
        
    except Exception as e:
        print(f"Error processing runs: {e}")
        return None

if __name__ == "__main__":
    print("🚀 Starting Model Evaluation...")
    result = compare_models()
    if result is not None:
        print("✅ Model evaluation completed successfully!")
        print("📈 Check the 'results/' folder for the comparison plot")
        print("🌐 Run 'mlflow ui' to view detailed results in browser")
    else:
        print("❌ Model evaluation failed. Please check if training was completed.")