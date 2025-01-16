import mlflow
print("Printing tracking URI scheme below") # When you first run mlflow code we will getting error which can be solved by below set_tracking_uri method
print(mlflow.get_tracking_uri())
print("\n")

mlflow.set_tracking_uri("http://127.0.0.1:5000")
print("Printing new tracking URI scheme below")
print(mlflow.get_tracking_uri())
print("\n")