import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import dagshub
dagshub.init(repo_owner='akshayparate61', repo_name='my-first-repo', mlflow=True) # setting remotely (This line of info will get from in dags conneted repo)
mlflow.set_tracking_uri("https://dagshub.com/akshayparate61/my-first-repo.mlflow") # link of our dags remote repo will get it from dags repo

# Load Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Define the params for RF model
max_depth = 8
n_estimators = 5

# Mention your experiment below
mlflow.set_experiment('YT-MLOPS-Exp1') # Give here experiment name which you have created in MLflow. or we can just give our exp name mlflow can automatic create exp like 'YT-MLOPS-Exp2'

with mlflow.start_run(): #experiment_id="from MLflow" put this into start_run func.
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('n_estimators', n_estimators)

    # print(accuracy)

    # Creating a confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')

    # save plot
    plt.savefig("Confusion-matrix.png")
    
    # log artifacts using mlflow
    mlflow.log_artifact("Confusion-matrix.png")
    mlflow.log_artifact(__file__) # Logging current file by this way "__file__" its a way
    
    # tags
    mlflow.set_tags({"Author": 'Akshay', "Project": "Wine Classification"})
    #
    # Log the model
    mlflow.sklearn.log_model(rf, "Random-Forest-Model")
    #
    print(accuracy)