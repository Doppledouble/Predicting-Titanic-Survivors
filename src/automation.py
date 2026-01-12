from src.Preprocess import TitanicPreprocessor
from typing import Dict, Tuple, Optional, Any, List
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score,  confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
import pandas as pd
from xgboost import XGBClassifier
import numpy as np

class ModelTrainer:
    """
    Wrapper class for training and evaluating classification models.
    """
    
    def __init__(self, model_type: str = 'logistic', **model_params):
        """
        Initialize the model trainer.
        
        Args:
            model_type: Type of model ('logistic', 'random_forest', 'xgboost', 'gradient_boosting')
            **model_params: Additional parameters for the model
        """
        self.model_type = model_type
        self.model_params = model_params
        self.model = self._initialize_model()
        self.is_fitted_ = False
        
    def _initialize_model(self):
        """Initialize the specified model type."""
        
        if self.model_type == 'logistic':
            return LogisticRegression(
                max_iter=1000,
                random_state=42,
                **self.model_params
            )
        
        elif self.model_type == 'random_forest':
            return RandomForestClassifier(
                **self.model_params
            )
            
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                **self.model_params
            )
            
        elif self.model_type == 'xgboost':
                return XGBClassifier(
                    **self.model_params
                )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> 'ModelTrainer':
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Self for method chaining
        """
        self.model.fit(X_train, y_train)
        self.is_fitted_ = True
        return self


    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted_:
            raise ValueError("Model must be trained before prediction.")
        return self.model.predict(X)
    
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_fitted_:
            raise ValueError("Model must be trained before prediction.")
        return self.model.predict_proba(X)[:, 1]
    
    
    def evaluate(self, 
                X: pd.DataFrame, 
                y: pd.Series,
                detailed: bool = False) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Features
            y: True labels
            detailed: If True, return detailed metrics
            
        Returns:
            Dictionary of metrics
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'roc_auc': roc_auc_score(y, y_proba),
            'f1': f1_score(y, y_pred),
        }
        
        if detailed:
            metrics.update({
                'precision': precision_score(y, y_pred),
                'recall': recall_score(y, y_pred),
            })
        
        return metrics
    
    def train_and_evaluate(self,
                          X_train: pd.DataFrame,
                          y_train: pd.Series,
                          X_val: pd.DataFrame,
                          y_val: pd.Series) -> Dict[str, Any]:
        """
        Train model and evaluate on both train and validation sets.
        
        Returns:
            Dictionary with train metrics, val metrics, and model
        """
        # Train the model
        self.train(X_train, y_train)
        
        # Evaluate on both sets
        train_metrics = self.evaluate(X_train, y_train, detailed=True)
        val_metrics = self.evaluate(X_val, y_val, detailed=True)
        
        # Check for overfitting
        overfit_score = train_metrics['accuracy'] - val_metrics['accuracy']
        
        return {
            'model': self.model,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'overfit_gap': overfit_score,
            'model_type': self.model_type
        }
        
        
    def cross_validate(self,
                      X: pd.DataFrame,
                      y: pd.Series,
                      cv: int = 5) -> Dict[str, Any]:
        """
        Perform cross-validation.
        
        Args:
            X: Features
            y: Target
            cv: Number of folds
            
        Returns:
            Dictionary with CV scores
        """
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        scores = cross_val_score(
            self.model, X, y, 
            cv=skf, 
            scoring='accuracy'
        )
        
        return {
            'cv_scores': scores,
            'cv_mean': scores.mean(),
            'cv_std': scores.std(),
            'cv_min': scores.min(),
            'cv_max': scores.max()
        }
    
    def get_feature_importance(self, 
                              feature_names: List[str],
                              top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance (for tree-based models).
        
        Args:
            feature_names: List of feature names
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if not hasattr(self.model, 'feature_importances_'):
            print(f"Model type '{self.model_type}' doesn't support feature importance")
            return None
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)

def find_best_parameter(X_train : pd.DataFrame,
                        y_train: pd.Series,
                        params: Dict[str, Any],
                        model_type: str,
                        cv : int = 5,
                        ) -> Dict[str, Any]:
    
    """
    Get the best parameters from a model using GridSearcCV

    Args:
        X_train : training Features
        y_train : training Labels
        params : Dictionary of model's modifiable parameters
        model_type : model name/type

    Returns:
        Dictionary of GridSearchCV results (best_params, best_score, best_estimator, cv_results)
        
    """
    trainer = ModelTrainer(model_type=model_type)
    
    grid_search = GridSearchCV(estimator=trainer.model, param_grid=params, cv=cv, verbose=1)
    grid_search.fit(X_train, y_train)
    
    print("Best parameters found: ", grid_search.best_params_)
    print("Best cross-validation score: ", grid_search.best_score_)
    
    return {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'best_estimator': grid_search.best_estimator_,
        'cv_results': grid_search.cv_results_
    }

def compare_models(X_train: pd.DataFrame,
                  y_train: pd.Series,
                  X_val: pd.DataFrame,
                  y_val: pd.Series,
                  models: Optional[Dict[str, Any]] = None,
                  grid_search: bool = False,
                  grid_search_cv: int = 5) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Compare multiple models on the same dataset.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        models: Dictionary of model configurations
        grid_search: Whether to perform grid search for best parameters
        grid_search_cv: Number of CV folds for grid search
        
    Returns:
        Tuple of (results_df, best_model_info) where:
        - results_df: DataFrame with comparison results
        - best_model_info: Dict with best model's name, type, and parameters
    """
    if models is None:
        # Default model configurations with parameter grids
        models = {
            'Logistic Regression': {
                'type': 'logistic',
                'params': {'C': 0.9, 'penalty': 'l2', 'solver': 'liblinear'},
                'param_grid': {
                    'C': [0.1, 0.5, 0.9, 1.0, 5.0],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear']
                }
            },
            'Random Forest': {
                'type': 'random_forest',
                'params': {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5},
                'param_grid': {
                    'n_estimators': [50, 100, 200, 400, 800],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10]
                }
            },
            'Gradient Boosting': {
                'type': 'gradient_boosting',
                'params': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5},
                'param_grid': {
                    'n_estimators': [50, 100, 200, 400, 800],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'max_depth': [2, 4, 6, 8, 10]
                }
            },
            'XGBoost': {
                'type': 'xgboost',
                'params': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5},
                'param_grid': {
                    'n_estimators': [50, 100, 200, 400, 800],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'max_depth': [2, 4, 6, 8, 10]
                }
            }
        }
    
    results = []
    model_params_dict = {}  # Store parameters for each model
    
    print("Training and comparing models...")
    print("=" * 80)
    
    for name, config in models.items():
        print(f"\nTraining {name}...")
        
        try:
            # Perform grid search if requested
            if grid_search and 'param_grid' in config:
                print(f"  Performing grid search for {name}...")
                grid_results = find_best_parameter(
                    X_train=X_train,
                    y_train=y_train,
                    params=config['param_grid'],
                    model_type=config['type'],
                    cv=grid_search_cv
                )
                
                # Use best parameters from grid search
                best_params = grid_results['best_params']
                print(f"  Using best parameters: {best_params}")
            else:
                # Use default parameters
                best_params = config.get('params', {})
            
            # Store model type and parameters
            model_params_dict[name] = {
                'type': config['type'],
                'params': best_params
            }
            
            # Initialize and train with best parameters
            trainer = ModelTrainer(
                model_type=config['type'],
                **best_params
            )
            print(f"  Model parameters: {trainer.model_params}")
            
            result = trainer.train_and_evaluate(X_train, y_train, X_val, y_val)
            
            # Build result row
            row = {
                'Model': name,
                'Train_Accuracy': result['train_metrics']['accuracy'],
                'Val_Accuracy': result['val_metrics']['accuracy'],
                'Val_ROC_AUC': result['val_metrics']['roc_auc'],
                'Val_F1': result['val_metrics']['f1'],
                'Val_Precision': result['val_metrics']['precision'],
                'Val_Recall': result['val_metrics']['recall'],
                'Overfit_Gap': result['overfit_gap']
            }
            
            # Add grid search score if performed
            if grid_search and 'param_grid' in config:
                row['GridSearch_Score'] = grid_results['best_score']
            
            results.append(row)
            
            print(f"  {name} - Val Accuracy: {row['Val_Accuracy']:.4f}, "
                  f"ROC-AUC: {row['Val_ROC_AUC']:.4f}, "
                  f"GridSearch Score: {row['GridSearch_Score']:.4f}")
            
        except Exception as e:
            print(f"  {name} failed: {str(e)}")
            continue
    
    print("\n" + "=" * 80)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by validation accuracy
    if len(results_df) > 0:
        results_df = results_df.sort_values('Val_Accuracy', ascending=False)
        
        # Get best model info
        best_model_name = results_df.iloc[0]['Model']
        best_model_info = {
            'name': best_model_name,
            'type': model_params_dict[best_model_name]['type'],
            'params': model_params_dict[best_model_name]['params']
        }
    else:
        best_model_info = None
    
    return results_df, best_model_info


def print_model_summary(results_df: pd.DataFrame, grid_search: bool = False):
    """
    Print a formatted summary of model comparison results.
    
    Args:
        results_df: DataFrame from compare_models()
        grid_search: Whether grid search was performed
    """
    print("\n" + "=" * 80)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 80)
    
    # Key metrics
    print("\n Validation Performance:")
    print(results_df[['Model', 'Val_Accuracy', 'Val_ROC_AUC', 'Val_F1']].to_string(index=False))
    
    # Grid search scores if available
    if grid_search and 'GridSearch_Score' in results_df.columns:
        print("\n Grid Search CV Scores:")
        print(results_df[['Model', 'GridSearch_Score']].to_string(index=False))
    
    print("\n  Overfitting Check:")
    print(results_df[['Model', 'Train_Accuracy', 'Val_Accuracy', 'Overfit_Gap']].to_string(index=False))
    
    # Best model
    best_model = results_df.iloc[0]
    print(f"\n Best Model: {best_model['Model']}")
    print(f"   Validation Accuracy: {best_model['Val_Accuracy']:.4f}")
    print(f"   ROC-AUC: {best_model['Val_ROC_AUC']:.4f}")
    print(f"   F1 Score: {best_model['Val_F1']:.4f}")
    print(f"   Overfitting Gap: {best_model['Overfit_Gap']:.4f}")
    
    if grid_search and 'GridSearch_Score' in results_df.columns:
        print(f"   Grid Search CV Score: {best_model['GridSearch_Score']:.4f}")
    
    # Warning for overfitting
    if best_model['Overfit_Gap'] > 0.05:
        print(f"\n  Warning: High overfitting gap ({best_model['Overfit_Gap']:.4f})")
        print("   Consider: regularization, reducing model complexity, or more data")
    
    # Additional insights
    print("\n Additional Insights:")
    
    # Check if any model has good balance
    balanced_models = results_df[results_df['Overfit_Gap'] <= 0.05]
    if len(balanced_models) > 0:
        print(f"    {len(balanced_models)} model(s) show good generalization (gap <= 0.05)")
    else:
        print(f"    All models show signs of overfitting")
    
    # Performance range
    acc_range = results_df['Val_Accuracy'].max() - results_df['Val_Accuracy'].min()
    print(f"   • Accuracy range across models: {acc_range:.4f}")
    
    print("\n" + "=" * 80)

def generate_submission(model: Any,
                       X_test: pd.DataFrame,
                       test_ids: pd.Series,
                       filename: str = 'submission.csv'):
    """
    Generate Kaggle submission file.
    
    Args:
        model: Trained model
        X_test: Test features
        test_ids: PassengerId from test set
        filename: Output filename
    """
    predictions = model.predict(X_test)
    
    submission = pd.DataFrame({
        'PassengerId': test_ids,
        'Survived': predictions
    })
    
    submission.to_csv(filename, index=False)
    print(f"\n Submission saved to {filename}")
    print(f"  Predicted survival rate: {predictions.mean():.2%}")
    print(f"  Total predictions: {len(predictions)}")


if __name__ == "__main__":
    print("Model Training Module")
    print("=" * 50)
    print("\nExample usage:")
    print("""
    # Single model training
    from model_trainer import ModelTrainer
    
    trainer = ModelTrainer(model_type='random_forest')
    results = trainer.train_and_evaluate(X_train, y_train, X_val, y_val)
    
    # Compare multiple models
    from model_trainer import compare_models, print_model_summary
    
    results_df = compare_models(X_train, y_train, X_val, y_val)
    print_model_summary(results_df)
    
    # Generate submission
    from model_trainer import generate_submission
    
    best_model = trainer.model
    generate_submission(best_model, X_test, test_ids)
    """)
    