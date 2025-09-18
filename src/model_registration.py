import mlflow
from mlflow.tracking import MlflowClient

def register_best_model():
    """Register the best performing model"""
    client = MlflowClient()
    
    try:
        # Get experiment
        experiment = mlflow.get_experiment_by_name("Wine_Classification_Experiment")
        if experiment is None:
            print("Experiment not found!")
            return None, None
        
        # Get all runs and find the best one
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        
        if runs.empty:
            print("No runs found!")
            return None, None
        
        # Find the run with highest accuracy
        best_run = runs.loc[runs['metrics.accuracy'].idxmax()]
        best_run_id = best_run['run_id']
        best_accuracy = best_run['metrics.accuracy']
        model_name = best_run['tags.mlflow.runName'] if 'tags.mlflow.runName' in best_run else 'Unknown'
        
        print(f"Best model: {model_name} with accuracy: {best_accuracy:.4f}")
        
        # Register the model (without description parameter)
        model_uri = f"runs:/{best_run_id}/model"
        registered_model_name = "Wine_Classification_Best_Model"
        
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=registered_model_name
        )
        
        print(f"Model registered successfully!")
        print(f"Model Name: {registered_model_name}")
        print(f"Version: {model_version.version}")
        
        return registered_model_name, model_version.version
        
    except Exception as e:
        print(f"Error registering model: {str(e)}")
        return None, None

if __name__ == "__main__":
    register_best_model()