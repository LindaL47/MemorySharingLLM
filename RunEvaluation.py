import os
import sys
import importlib

# Definition of test files for each domain
# These match the calls found in the respective Integrate.py files
tasks = {
    "Literal Creation": [
        "limerickTest.jsonl", 
        "lvshiTest.jsonl", 
        "sonnetTest.jsonl"
    ],
    "Logic Problem_Solving": [
        "puzzleTest.jsonl", 
        "punsTest.jsonl", 
        "riddlesTest.jsonl"
    ],
    "Plan Generation": [
        "fitnessTest.jsonl", 
        "studyTest.jsonl", 
        "travelTest.jsonl"
    ],
    "OnePool": [
        "lvshiTest.jsonl", "limerickTest.jsonl", "sonnetTest.jsonl",
        "punsTest.jsonl", "puzzleTest.jsonl", "riddlesTest.jsonl",
        "fitnessTest.jsonl", "studyTest.jsonl", "travelTest.jsonl"
    ]
}

def run_evaluation(directory, test_files):
    print(f"\n{'='*60}")
    print(f"Starting Evaluation for Directory: {directory}")
    print(f"{'='*60}")
    
    base_path = os.getcwd()
    target_path = os.path.join(base_path, directory)
    
    if not os.path.exists(target_path):
        print(f"Error: Directory not found: {target_path}")
        return

    # Check for model.pth (Required for retrieval)
    model_path = os.path.join(target_path, "model.pth")
    if not os.path.exists(model_path):
        print(f"Warning: 'model.pth' not found in {directory}.")
        print("Evaluation requires a trained model. Please run 'Integrate.py' in this directory first.")
        return

    # Switch working directory to handle relative imports and file reads in Evaluate.py
    os.chdir(target_path)
    # Add current directory to sys.path to allow imports
    if os.getcwd() not in sys.path:
        sys.path.append(os.getcwd())
    
    try:
        # Dynamically import Evaluate module
        import Evaluate
        importlib.reload(Evaluate) # Ensure we get a fresh instance if run multiple times
        
        # Determine the correct pool file
        pool_file = ""
        if "Literal" in directory: pool_file = "LiteralPool.jsonl"
        elif "Logic" in directory: pool_file = "LogicPool.jsonl"
        elif "Plan" in directory: pool_file = "PlanPool.jsonl"
        elif "OnePool" in directory: pool_file = "SinglePool.jsonl"
        
        if not os.path.exists(pool_file):
            print(f"Error: Memory pool file '{pool_file}' not found.")
            return

        for test_file in test_files:
            print(f"\n[Evaluating {test_file}]")
            if os.path.exists(test_file):
                print(f"Running Evaluate.main('{pool_file}', '{test_file}')...")
                # Evaluate.main prints:
                # 1. ROUGE score (Aggregated F1)
                # 2. BERTScore (Precision)
                Evaluate.main(pool_file, test_file)
            else:
                print(f"Skipping: Test file '{test_file}' not found.")
                
    except ImportError as e:
        print(f"Import Error: {e}")
    except Exception as e:
        print(f"Runtime Error: {e}")
    finally:
        # Restore environment
        if os.getcwd() in sys.path:
            sys.path.remove(os.getcwd())
        os.chdir(base_path)

if __name__ == "__main__":
    print("This script runs the evaluation metrics (ROUGE and BERTScore) used in the paper.")
    print("It checks for 'model.pth' in each directory. If missing, you must run 'Integrate.py' first.\n")

    # Uncomment the sections you want to run. 
    # Currently only 'Literal Creation' has a pre-existing model.pth based on file scan.
    
    run_evaluation("Literal Creation", tasks["Literal Creation"])
    
    # These will likely fail if Integrate.py hasn't been run yet to generate model.pth
    # run_evaluation("Logic Problem_Solving", tasks["Logic Problem_Solving"])
    # run_evaluation("Plan Generation", tasks["Plan Generation"])
    # run_evaluation("OnePool", tasks["OnePool"])
