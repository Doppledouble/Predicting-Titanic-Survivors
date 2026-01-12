from Preprocess import TitanicPreprocessor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Load your data
train_data = pd.read_csv("Pre_processed_train.csv")
X = train_data.drop(columns=['Survived'])
y = train_data['Survived']

test_data = pd.read_csv("Pre_processed_test.csv")

## Split for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

## Train Logistic Regression  model
def train_LogReg(
        X_train : pd.DataFrame,
        y_train : pd.Series,
        **kwargs) -> LogisticRegression:
    
    model = LogisticRegression(**kwargs)
    model.fit(X_train,y_train)
    
    return model

def train_RandForest(
        X_train : pd.DataFrame,
        y_train : pd.Series,
        **kwargs) -> RandomForestClassifier:
    
    model = RandomForestClassifier(**kwargs)
    model.fit(X_train,y_train)

    return model


Random_forest = train_RandForest(
    X_train,
    y_train, 
    n_estimators=100,
    min_samples_split=5,
    random_state=42
    )


predictions = Random_forest.predict(X_val)
print(f"Accuracy : {accuracy_score(y_val, predictions):.5f}")


final_predictions = Random_forest.predict(test_data)
submission = pd.DataFrame({
     "PassengerId": test_data["PassengerId"],
     "Survived": final_predictions
 })

submission.to_csv("submission.csv", index=False)
print("submission.csv saved!")
print(submission["Survived"].value_counts())


## Initialize and fit preprocessor
# preprocessor = TitanicPreprocessor()
# X, y = preprocessor.fit_transform(train_data)

## For test data
# X_test = preprocessor.transform(test_data)



# Save the new datasest instead of reprocess it again
# X['Survived'] = y
# X .to_csv('Pre_processed_train.csv', index=False)

# X_test.to_csv('Pre_processed_test.csv', index=False)