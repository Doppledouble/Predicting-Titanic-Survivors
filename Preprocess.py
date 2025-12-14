"""
Titanic Dataset Preprocessing Module

A modular preprocessing pipeline for Titanic dataset that can be reused
across different machine learning models.

Usage:
    from titanic_preprocessor import TitanicPreprocessor
    
    preprocessor = TitanicPreprocessor()
    X_train, y_train = preprocessor.fit_transform(train_data)
    X_test = preprocessor.transform(test_data)
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class TitanicPreprocessor:
    """
    Preprocessing pipeline for Titanic dataset with feature engineering.
    
    This class handles:
    - Name parsing (first/last name, title extraction)
    - Age imputation using grouped medians
    - Ticket feature engineering
    - Cabin feature extraction
    - One-hot encoding of categorical variables
    - Feature selection
    """
    
    def __init__(self, 
                 rare_titles: Optional[List[str]] = None,
                 rare_ticket_prefixes: Optional[List[str]] = None):
        """
        Initialize the preprocessor.
        
        Args:
            rare_titles: List of rare titles to group together. If None, uses default.
            rare_ticket_prefixes: List of rare ticket prefixes to group. If None, uses default.
        """
        self.rare_titles = rare_titles or [
            'Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 
            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'
        ]
        
        self.rare_ticket_prefixes = rare_ticket_prefixes or [
            'A5', 'A4', 'AQ3', 'AQ4', 'AS', 'C', 'CA', 'CASOTON', 'FC', 'FCC',
            'FA', 'LP', 'PP', 'PPP', 'SC', 'SCA3', 'SCA4', 'SCAH', 'SCOW',
            'SOP', 'SOPP', 'SOTONO', 'SP', 'STONO', 'SWPP', 'WEP', 'WC',
            'A', 'SOTONOQ', 'STONO2', 'SOTONO2', 'STONOQ', 'SCPARIS', 'SOC'
        ]
        
        # Store fitted attributes
        self.fitted_columns_ = None
        self.is_fitted_ = False
        
    def _parse_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse Name column into Last_name and First_name.
        
        Args:
            df: Input dataframe with 'Name' column
            
        Returns:
            DataFrame with added Last_name and First_name columns
        """
        df = df.copy()
        df[['Last_name', 'First_name']] = df['Name'].str.split(', ', n=1, expand=True)
        df['Last_name'] = df['Last_name'].str.strip()
        df['First_name'] = df['First_name'].str.strip()
        return df
    
    def _extract_title(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract title from First_name and clean rare titles.
        
        Args:
            df: DataFrame with 'First_name' column
            
        Returns:
            DataFrame with added 'Title' column
        """
        df = df.copy()
        df['Title'] = df['First_name'].str.extract('([A-Za-z]+)\.', expand=False)
        
        # Standardize similar titles
        df['Title'] = df['Title'].replace({
            'Mlle': 'Miss', 
            'Ms': 'Miss', 
            'Mme': 'Mrs'
        })
        
        # Group rare titles
        df['Title'] = df['Title'].replace(self.rare_titles, 'Rare')
        
        return df
    
    def _impute_age(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing Age values using hierarchical group medians.
        
        Strategy:
        1. First try: Pclass + Sex + Title
        2. Fallback: Sex + Title
        3. Final fallback: Title only
        
        Args:
            df: DataFrame with Age column
            
        Returns:
            DataFrame with imputed Age values
        """
        df = df.copy()
        
        # Level 1: Group by Pclass, Sex, and Title
        df['Age'] = df.groupby(['Pclass', 'Sex_male', 'Title'])['Age'].transform(
            lambda x: x.fillna(x.median())
        )
        
        # Level 2: Group by Sex and Title (for remaining nulls)
        df['Age'] = df.groupby(['Sex_male', 'Title'])['Age'].transform(
            lambda x: x.fillna(x.median())
        )
        
        # Level 3: Group by Title only (final fallback)
        df['Age'] = df.groupby(['Title'])['Age'].transform(
            lambda x: x.fillna(x.median())
        )
        
        return df
    
    def _engineer_ticket_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from Ticket column.
        
        Features created:
        - Ticket_prefix: Cleaned prefix of ticket
        - Ticket_len: Length of ticket string
        - Ticket_numeric: Whether ticket is purely numeric
        
        Args:
            df: DataFrame with 'Ticket' column
            
        Returns:
            DataFrame with ticket features
        """
        df = df.copy()
        
        # Extract ticket prefix
        df['Ticket_prefix'] = df['Ticket'].apply(
            lambda x: x.split()[0] if len(x.split()) > 1 else 'None'
        )
        
        # Clean prefix
        df['Ticket_prefix'] = (df['Ticket_prefix']
                               .str.replace('.', '', regex=False)
                               .str.replace('/', '', regex=False)
                               .str.upper())
        
        # Group rare prefixes
        df['Ticket_prefix'] = df['Ticket_prefix'].replace(
            self.rare_ticket_prefixes, 'Rare'
        )
        
        # Ticket length (powerful feature)
        df['Ticket_len'] = df['Ticket'].apply(
            lambda x: len(x.replace(' ', '').replace('.', '').replace('/', ''))
        )
        
        # Purely numeric ticket indicator
        df['Ticket_numeric'] = df['Ticket'].apply(
            lambda x: 1 if x.replace(' ', '').isdigit() else 0
        )
        
        return df
    
    def _engineer_cabin_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create binary feature indicating cabin presence.
        
        Args:
            df: DataFrame with 'Cabin' column
            
        Returns:
            DataFrame with HasCabin feature
        """
        df = df.copy()
        df['HasCabin'] = df['Cabin'].notna().astype(int)
        return df
    
    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        One-hot encode categorical variables.
        
        NOTE: This method is deprecated in favor of encoding steps
        done separately in fit_transform and transform methods.
        Kept for backward compatibility.
        
        Args:
            df: DataFrame with categorical columns
            
        Returns:
            DataFrame with encoded categories
        """
        df = df.copy()
        
        # Encode Sex and Embarked
        if 'Sex' in df.columns:
            df = pd.get_dummies(df, columns=['Sex'], dtype=int)
        if 'Embarked' in df.columns:
            df = pd.get_dummies(df, columns=['Embarked'], dtype=int)
            
        # Encode Title and Ticket_prefix
        if 'Title' in df.columns:
            df = pd.get_dummies(df, columns=['Title'], dtype=int)
        if 'Ticket_prefix' in df.columns:
            df = pd.get_dummies(df, columns=['Ticket_prefix'], dtype=int)
            
        return df
    
    def _drop_unnecessary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop columns not needed for modeling.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with unnecessary columns removed
        """
        cols_to_drop = [
            'Name', 'Cabin', 'Ticket', 'Sex_female', 
            'First_name', 'Last_name'
        ]
        
        df = df.drop(columns=cols_to_drop, errors='ignore')
        return df
    
    def _align_columns(self, df: pd.DataFrame, reference_cols: List[str]) -> pd.DataFrame:
        """
        Align columns with reference set (for test data).
        
        Args:
            df: Input DataFrame
            reference_cols: List of expected column names
            
        Returns:
            DataFrame with aligned columns
        """
        # Add missing columns with 0s
        for col in reference_cols:
            if col not in df.columns:
                df[col] = 0
        
        # Keep only reference columns (in same order)
        df = df[reference_cols]
        
        return df
    
    def fit_transform(self, 
                     df: pd.DataFrame, 
                     target_col: str = 'Survived') -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Fit the preprocessor and transform training data.
        
        Args:
            df: Training dataframe
            target_col: Name of target column (default: 'Survived')
            
        Returns:
            Tuple of (X_transformed, y) where y is the target if present
        """
        df = df.copy()
        
        # Step 1: Parse names
        df = self._parse_names(df)
        
        # Step 2: Extract title (needs First_name)
        df = self._extract_title(df)
        
        # Step 3: Encode Sex and Embarked only (keep Title for imputation)
        if 'Sex' in df.columns:
            df = pd.get_dummies(df, columns=['Sex'], dtype=int)
        if 'Embarked' in df.columns:
            df = pd.get_dummies(df, columns=['Embarked'], dtype=int)
        
        # Step 4: Impute age (needs encoded Sex and original Title column)
        df = self._impute_age(df)
        
        # Step 5: Engineer ticket features
        df = self._engineer_ticket_features(df)
        
        # Step 6: Engineer cabin features
        df = self._engineer_cabin_features(df)
        
        # Step 7: Now encode Title and Ticket_prefix
        if 'Title' in df.columns:
            df = pd.get_dummies(df, columns=['Title'], dtype=int)
        if 'Ticket_prefix' in df.columns:
            df = pd.get_dummies(df, columns=['Ticket_prefix'], dtype=int)
        
        # Step 8: Drop unnecessary columns
        df = self._drop_unnecessary_columns(df)
        
        # Step 9: Handle Fare missing values
        if 'Fare' in df.columns:
            df['Fare'] = df['Fare'].fillna(df['Fare'].median())
        
        # Separate target if present
        y = None
        if target_col in df.columns:
            y = df[target_col].copy()
            df = df.drop(columns=[target_col])
        
        # Store column names for later alignment
        self.fitted_columns_ = df.columns.tolist()
        self.is_fitted_ = True
        
        return df, y
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform test/new data using fitted preprocessor.
        
        Args:
            df: Test dataframe
            
        Returns:
            Transformed dataframe
            
        Raises:
            ValueError: If preprocessor hasn't been fitted yet
        """
        if not self.is_fitted_:
            raise ValueError("Preprocessor must be fitted before transform. Use fit_transform() first.")
        
        df = df.copy()
        
        # Apply same transformation steps
        df = self._parse_names(df)
        df = self._extract_title(df)
        
        # Encode Sex and Embarked only
        if 'Sex' in df.columns:
            df = pd.get_dummies(df, columns=['Sex'], dtype=int)
        if 'Embarked' in df.columns:
            df = pd.get_dummies(df, columns=['Embarked'], dtype=int)
            
        df = self._impute_age(df)
        df = self._engineer_ticket_features(df)
        df = self._engineer_cabin_features(df)
        
        # Encode Title and Ticket_prefix
        if 'Title' in df.columns:
            df = pd.get_dummies(df, columns=['Title'], dtype=int)
        if 'Ticket_prefix' in df.columns:
            df = pd.get_dummies(df, columns=['Ticket_prefix'], dtype=int)
            
        df = self._drop_unnecessary_columns(df)
        
        # Handle Fare
        if 'Fare' in df.columns:
            df['Fare'] = df['Fare'].fillna(df['Fare'].median())
        
        # Remove target if present
        if 'Survived' in df.columns:
            df = df.drop(columns=['Survived'])
        
        # Align columns with training data
        df = self._align_columns(df, self.fitted_columns_)
        
        return df
    
    def fit(self, df: pd.DataFrame, target_col: str = 'Survived'):
        """
        Fit the preprocessor without transforming (useful for pipeline compatibility).
        
        Args:
            df: Training dataframe
            target_col: Name of target column
        """
        self.fit_transform(df, target_col)
        return self


# Convenience function for quick usage
def preprocess_titanic(train_df: pd.DataFrame, 
                       test_df: Optional[pd.DataFrame] = None,
                       target_col: str = 'Survived') -> Tuple:
    """
    Quick preprocessing function for Titanic data.
    
    Args:
        train_df: Training dataframe
        test_df: Optional test dataframe
        target_col: Name of target column
        
    Returns:
        If test_df is None: (X_train, y_train)
        If test_df provided: (X_train, y_train, X_test)
    """
    preprocessor = TitanicPreprocessor()
    X_train, y_train = preprocessor.fit_transform(train_df, target_col)
    
    if test_df is not None:
        X_test = preprocessor.transform(test_df)
        return X_train, y_train, X_test
    
    return X_train, y_train


if __name__ == "__main__":
    # Example usage
    print("Titanic Preprocessor Module")
    print("="*50)
    print("\nExample usage:")
    print("""
    # Method 1: Using the class
    from titanic_preprocessor import TitanicPreprocessor
    
    preprocessor = TitanicPreprocessor()
    X_train, y_train = preprocessor.fit_transform(train_data)
    X_test = preprocessor.transform(test_data)
    
    # Method 2: Quick function
    from titanic_preprocessor import preprocess_titanic
    
    X_train, y_train, X_test = preprocess_titanic(train_data, test_data)
    """)