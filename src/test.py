from src.automation import ModelTrainer, compare_models, print_model_summary, generate_submission
from sklearn.model_selection import train_test_split
import pandas as pd

train_data = pd.read_csv("Pre_processed_train.csv")
X = train_data.drop(columns=['Survived','PassengerId'])
y = train_data['Survived']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Compare multiple models
results_df, best_model_info = compare_models(X_train, y_train, X_val, y_val,grid_search=True)
print_model_summary(results_df)


print(f"Best Model: {best_model_info['name']}")
print(f"Model Type: {best_model_info['type']}")
print(f"Parameters: {best_model_info['params']}")

trainer = ModelTrainer(
    model_type=best_model_info['type'],
    **best_model_info['params']
)


print(trainer.model_params)

trainer.train(X_train,y_train)
print(f"Training Model {trainer.model_type} is done!")

### Generate Submission
test_dataset = pd.read_csv("Pre_processed_test.csv")

X_test = test_dataset.drop(columns=["PassengerId"])
test_ids = test_dataset["PassengerId"]

generate_submission(trainer.model, X_test, test_ids)