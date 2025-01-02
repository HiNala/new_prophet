import shutil
from pathlib import Path

def cleanup_project():
    """Clean up project structure by removing duplicate directories."""
    # Directories to remove
    dirs_to_remove = [
        'pipeline',
        'cli',
        'tests',
        'prophet_forecaster/prophet_forecaster'  # Nested directory
    ]
    
    # Remove directories
    for dir_path in dirs_to_remove:
        path = Path(dir_path)
        if path.exists():
            print(f"Removing {path}")
            shutil.rmtree(path)
    
    # Ensure correct directory structure exists
    dirs_to_create = [
        'prophet_forecaster/pipeline/data_ingestion/fetchers',
        'prophet_forecaster/pipeline/data_visualization',
        'prophet_forecaster/pipeline/data_analysis',
        'prophet_forecaster/tests/unit',
        'prophet_forecaster/tests/integration',
        'prophet_forecaster/tests/e2e',
        'data/development/raw',
        'data/development/processed',
        'data/development/visualizations',
        'data/development/reports',
        'data/test/raw',
        'data/test/processed',
        'data/test/visualizations',
        'data/test/reports'
    ]
    
    # Create directories
    for dir_path in dirs_to_create:
        path = Path(dir_path)
        if not path.exists():
            print(f"Creating {path}")
            path.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    cleanup_project() 