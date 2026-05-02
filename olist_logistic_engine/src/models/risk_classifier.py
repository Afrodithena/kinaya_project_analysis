"""
Risk classification model for route delay prediction
Uses Random Forest classifier to predict route performance risk
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from typing import Dict, Any, Optional, Tuple
from pathlib import Path


class RouteRiskClassifier:
    """
    Classifier for route delivery risk prediction.
    Predicts whether a route will be Slow, Normal, or Fast.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the risk classifier.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.is_fitted = False
    
    def preprocess_data(self, df: pd.DataFrame, fit_encoders: bool = True) -> np.ndarray:
        """
        Preprocess route data for model input.
        
        Args:
            df: DataFrame with route features
            fit_encoders: Whether to fit label encoders (True for training, False for prediction)
        
        Returns:
            Preprocessed feature matrix
        """
        df = df.copy()
        
        # Define categorical columns
        categorical_cols = ['seller_state', 'customer_state', 'seller_region', 'customer_region']
        
        # Define numerical columns
        numerical_cols = [
            'distance_km', 'order_count', 'avg_price', 'avg_freight',
            'freight_per_km', 'revenue_per_order', 'freight_to_price_ratio'
        ]
        
        # Handle missing columns
        available_numerical = [col for col in numerical_cols if col in df.columns]
        available_categorical = [col for col in categorical_cols if col in df.columns]
        
        # Process categorical columns
        categorical_encoded = []
        for col in available_categorical:
            if fit_encoders:
                encoder = LabelEncoder()
                # Handle unseen values by filling with most frequent
                df[col] = df[col].fillna('Unknown')
                encoded = encoder.fit_transform(df[col])
                self.label_encoders[col] = encoder
            else:
                encoder = self.label_encoders.get(col)
                if encoder:
                    # Handle unseen categories
                    df[col] = df[col].fillna('Unknown')
                    df[col] = df[col].apply(lambda x: x if x in encoder.classes_ else 'Unknown')
                    encoded = encoder.transform(df[col])
                else:
                    encoded = np.zeros(len(df))
            categorical_encoded.append(encoded.reshape(-1, 1))
        
        # Process numerical columns
        numerical_data = df[available_numerical].fillna(df[available_numerical].median())
        
        # Combine features
        if categorical_encoded:
            categorical_matrix = np.hstack(categorical_encoded)
            feature_matrix = np.hstack([numerical_data.values, categorical_matrix])
        else:
            feature_matrix = numerical_data.values
        
        # Scale numerical features only
        if fit_encoders:
            self.feature_columns = available_numerical + available_categorical
            numerical_scaled = self.scaler.fit_transform(numerical_data)
        else:
            numerical_scaled = self.scaler.transform(numerical_data)
        
        # Recombine with categorical (categorical not scaled)
        if categorical_encoded:
            final_matrix = np.hstack([numerical_scaled, categorical_matrix])
        else:
            final_matrix = numerical_scaled
        
        return final_matrix
    
    def train(self, df: pd.DataFrame, target_col: str = 'performance') -> Dict[str, Any]:
        """
        Train the risk classifier model.
        
        Args:
            df: DataFrame with features and target column
            target_col: Name of the target column
        
        Returns:
            Dictionary with training metrics
        """
        # Prepare target
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(df[target_col])
        self.target_classes = target_encoder.classes_.tolist()
        
        # Preprocess features
        X = self.preprocess_data(df, fit_encoders=True)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, target_names=self.target_classes),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'train_score': self.model.score(X_train, y_train),
            'test_score': self.model.score(X_test, y_test)
        }
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=5)
        metrics['cv_mean'] = cv_scores.mean()
        metrics['cv_std'] = cv_scores.std()
        
        self.is_fitted = True
        return metrics
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict risk category for routes.
        
        Args:
            df: DataFrame with route features
        
        Returns:
            Array of predictions
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call train() first.")
        
        X = self.preprocess_data(df, fit_encoders=False)
        predictions = self.model.predict(X)
        
        return predictions
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Get probability scores for each risk category.
        
        Args:
            df: DataFrame with route features
        
        Returns:
            Array of probability scores
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call train() first.")
        
        X = self.preprocess_data(df, fit_encoders=False)
        probabilities = self.model.predict_proba(X)
        
        return probabilities
    
    def get_feature_importance(self, feature_names: Optional[list] = None) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Returns:
            DataFrame with feature names and importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call train() first.")
        
        if feature_names is None:
            feature_names = self.feature_columns
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, filepath: str):
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'target_classes': self.target_classes,
            'random_state': self.random_state,
            'is_fitted': self.is_fitted
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_columns = model_data['feature_columns']
        self.target_classes = model_data['target_classes']
        self.random_state = model_data['random_state']
        self.is_fitted = model_data['is_fitted']
        print(f"Model loaded from {filepath}")


def predict_route_risk(df: pd.DataFrame, model_path: str) -> pd.DataFrame:
    """
    Convenience function to predict route risk using saved model.
    
    Args:
        df: DataFrame with route features
        model_path: Path to saved model file
    
    Returns:
        DataFrame with predictions added
    """
    classifier = RouteRiskClassifier()
    classifier.load_model(model_path)
    
    predictions = classifier.predict(df)
    probabilities = classifier.predict_proba(df)
    
    df = df.copy()
    df['risk_prediction'] = [classifier.target_classes[p] for p in predictions]
    
    # Add probability columns
    for i, class_name in enumerate(classifier.target_classes):
        df[f'prob_{class_name}'] = probabilities[:, i]
    
    return df


def calculate_risk_score(df: pd.DataFrame, model_path: Optional[str] = None) -> pd.DataFrame:
    """
    Calculate risk score (0-100) for routes based on probabilities.
    
    Args:
        df: DataFrame with route features or predictions
        model_path: Path to model (optional, if predictions already exist)
    
    Returns:
        DataFrame with risk_score column
    """
    df = df.copy()
    
    if 'prob_Critical' in df.columns:
        # Use existing probability
        df['risk_score'] = df['prob_Critical'] * 100
    elif 'prob_Slow' in df.columns:
        df['risk_score'] = df['prob_Slow'] * 80 + df.get('prob_Critical', 0) * 100
    elif model_path:
        # Load model and predict
        classifier = RouteRiskClassifier()
        classifier.load_model(model_path)
        probabilities = classifier.predict_proba(df)
        
        # Get index of Critical class
        if 'Critical' in classifier.target_classes:
            critical_idx = classifier.target_classes.index('Critical')
            df['risk_score'] = probabilities[:, critical_idx] * 100
        else:
            df['risk_score'] = 50  # Default
    else:
        # Calculate based on delivery days
        if 'avg_delivery_days' in df.columns:
            max_days = df['avg_delivery_days'].max()
            df['risk_score'] = (df['avg_delivery_days'] / max_days) * 100
        else:
            df['risk_score'] = 50
    
    # Add risk level
    df['risk_level'] = pd.cut(
        df['risk_score'],
        bins=[0, 25, 50, 75, 101],
        labels=['Low', 'Medium-Low', 'Medium-High', 'High']
    )
    
    return df


if __name__ == "__main__":
    # Example usage
    print("RouteRiskClassifier module loaded")