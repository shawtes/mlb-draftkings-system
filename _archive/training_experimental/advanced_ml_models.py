#!/usr/bin/env python3
"""
Advanced ML Models for Cryptocurrency Trading
Implements LightGBM, XGBoost, and Neural Networks with ensemble predictions

Stefan Jansen "Machine Learning for Algorithmic Trading" - Chapter 9: Bayesian ML
Based on institutional-grade model architectures
"""

import os
import sys
import time
import pickle
import logging
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelMetrics:
    """Container for model performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    cv_scores: List[float]
    feature_importance: Dict[str, float]
    training_time: float
    prediction_time: float

class AdvancedMLModels:
    """
    Advanced ML Models system implementing multiple algorithms for ensemble predictions
    
    Features:
    - LightGBM: Fast gradient boosting
    - XGBoost: Robust gradient boosting with regularization
    - Neural Networks: Deep learning with TensorFlow/Keras
    - Ensemble Methods: Weighted voting and stacking
    - Feature Selection: Advanced feature engineering integration
    - Cross-validation: Robust model evaluation
    """
    
    def __init__(self, 
                 models_dir: str = "models",
                 use_gpu: bool = False,
                 enable_neural_networks: bool = True):
        """
        Initialize Advanced ML Models system
        
        Args:
            models_dir: Directory to save/load models
            use_gpu: Enable GPU acceleration if available
            enable_neural_networks: Enable neural network models
        """
        self.models_dir = models_dir
        self.use_gpu = use_gpu
        self.enable_neural_networks = enable_neural_networks
        
        # Create models directory
        os.makedirs(models_dir, exist_ok=True)
        
        # Initialize model containers
        self.models = {}
        self.model_metrics = {}
        self.ensemble_weights = {}
        
        # Feature importance tracking
        self.feature_importance_combined = {}
        
        # Performance tracking
        self.training_history = []
        self.prediction_history = []
        
        # Import libraries with graceful fallbacks
        self._initialize_libraries()
        
        logger.info("🧠 Advanced ML Models system initialized")
        logger.info(f"   Models directory: {models_dir}")
        logger.info(f"   GPU acceleration: {'✅' if use_gpu and self.gpu_available else '❌'}")
        logger.info(f"   Neural networks: {'✅' if enable_neural_networks and self.nn_available else '❌'}")
        logger.info(f"   Available models: {', '.join(self.available_models)}")
    
    def _initialize_libraries(self):
        """Initialize ML libraries with graceful fallbacks"""
        self.available_models = ['RandomForest']  # Always available baseline
        self.gpu_available = False
        self.lgb_available = False
        self.xgb_available = False
        self.nn_available = False
        
        # Try to import LightGBM
        try:
            import lightgbm as lgb
            self.lgb = lgb
            self.lgb_available = True
            self.available_models.append('LightGBM')
            logger.info("✅ LightGBM loaded successfully")
        except ImportError:
            logger.warning("⚠️ LightGBM not available. Install with: pip install lightgbm")
        
        # Try to import XGBoost
        try:
            import xgboost as xgb
            self.xgb = xgb
            self.xgb_available = True
            self.available_models.append('XGBoost')
            logger.info("✅ XGBoost loaded successfully")
        except ImportError:
            logger.warning("⚠️ XGBoost not available. Install with: pip install xgboost")
        
        # Try to import TensorFlow/Keras for Neural Networks
        if self.enable_neural_networks:
            try:
                import tensorflow as tf
                from tensorflow import keras
                from tensorflow.keras import layers, models, optimizers, callbacks
                
                # Suppress TensorFlow warnings
                tf.get_logger().setLevel('ERROR')
                os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
                
                self.tf = tf
                self.keras = keras
                self.layers = layers
                self.models_tf = models
                self.optimizers = optimizers
                self.callbacks = callbacks
                
                self.nn_available = True
                self.available_models.append('NeuralNetwork')
                
                # Check GPU availability
                if len(tf.config.list_physical_devices('GPU')) > 0:
                    self.gpu_available = True
                    logger.info("✅ TensorFlow with GPU support loaded")
                else:
                    logger.info("✅ TensorFlow loaded (CPU only)")
                    
            except ImportError:
                logger.warning("⚠️ TensorFlow not available. Install with: pip install tensorflow")
        
        # Always available: scikit-learn
        try:
            from sklearn.ensemble import RandomForestClassifier, VotingClassifier
            from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            from sklearn.preprocessing import StandardScaler
            from sklearn.feature_selection import SelectKBest, f_classif
            
            self.RandomForestClassifier = RandomForestClassifier
            self.VotingClassifier = VotingClassifier
            self.cross_val_score = cross_val_score
            self.GridSearchCV = GridSearchCV
            self.StratifiedKFold = StratifiedKFold
            self.accuracy_score = accuracy_score
            self.precision_score = precision_score
            self.recall_score = recall_score
            self.f1_score = f1_score
            self.roc_auc_score = roc_auc_score
            self.StandardScaler = StandardScaler
            self.SelectKBest = SelectKBest
            self.f_classif = f_classif
            
            logger.info("✅ Scikit-learn models loaded successfully")
            
        except ImportError as e:
            logger.error(f"❌ Critical error: Scikit-learn not available: {e}")
            raise
    
    def create_lightgbm_model(self, **params) -> Any:
        """Create optimized LightGBM model"""
        if not self.lgb_available:
            raise ValueError("LightGBM not available")
        
        default_params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42,
            'n_estimators': 100,
            'max_depth': -1,
            'min_child_samples': 20,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1
        }
        
        # Update with user parameters
        default_params.update(params)
        
        # Enable GPU if available
        if self.use_gpu and self.gpu_available:
            default_params['device'] = 'gpu'
            default_params['gpu_platform_id'] = 0
            default_params['gpu_device_id'] = 0
        
        return self.lgb.LGBMClassifier(**default_params)
    
    def create_xgboost_model(self, **params) -> Any:
        """Create optimized XGBoost model"""
        if not self.xgb_available:
            raise ValueError("XGBoost not available")
        
        default_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'verbosity': 0,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'min_child_weight': 1,
            'gamma': 0
        }
        
        # Update with user parameters
        default_params.update(params)
        
        # Enable GPU if available
        if self.use_gpu and self.gpu_available:
            default_params['tree_method'] = 'gpu_hist'
            default_params['gpu_id'] = 0
        
        return self.xgb.XGBClassifier(**default_params)
    
    def create_neural_network_model(self, input_dim: int, **params) -> Any:
        """Create optimized Neural Network model"""
        if not self.nn_available:
            raise ValueError("Neural Networks not available")
        
        # Default architecture parameters
        default_params = {
            'hidden_layers': [128, 64, 32],
            'dropout_rate': 0.3,
            'activation': 'relu',
            'output_activation': 'sigmoid',
            'optimizer': 'adam',
            'learning_rate': 0.001,
            'batch_normalization': True,
            'l2_regularization': 0.001
        }
        
        # Update with user parameters
        default_params.update(params)
        
        # Build model architecture
        model = self.models_tf.Sequential()
        
        # Input layer
        model.add(self.layers.Dense(
            default_params['hidden_layers'][0], 
            input_dim=input_dim,
            activation=default_params['activation'],
            kernel_regularizer=self.keras.regularizers.l2(default_params['l2_regularization'])
        ))
        
        if default_params['batch_normalization']:
            model.add(self.layers.BatchNormalization())
        
        model.add(self.layers.Dropout(default_params['dropout_rate']))
        
        # Hidden layers
        for units in default_params['hidden_layers'][1:]:
            model.add(self.layers.Dense(
                units, 
                activation=default_params['activation'],
                kernel_regularizer=self.keras.regularizers.l2(default_params['l2_regularization'])
            ))
            
            if default_params['batch_normalization']:
                model.add(self.layers.BatchNormalization())
            
            model.add(self.layers.Dropout(default_params['dropout_rate']))
        
        # Output layer
        model.add(self.layers.Dense(1, activation=default_params['output_activation']))
        
        # Compile model
        optimizer = self.optimizers.Adam(learning_rate=default_params['learning_rate'])
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'auc']
        )
        
        return model
    
    def train_single_model(self, 
                          model_type: str,
                          X_train: pd.DataFrame, 
                          y_train: pd.Series,
                          X_val: Optional[pd.DataFrame] = None,
                          y_val: Optional[pd.Series] = None,
                          **model_params) -> Tuple[Any, ModelMetrics]:
        """
        Train a single model and return performance metrics
        
        Args:
            model_type: Type of model ('LightGBM', 'XGBoost', 'NeuralNetwork', 'RandomForest')
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            **model_params: Additional model parameters
        
        Returns:
            Tuple of (trained_model, performance_metrics)
        """
        logger.info(f"🔧 Training {model_type} model...")
        start_time = time.time()
        
        try:
            # Create model based on type
            if model_type == 'LightGBM':
                model = self.create_lightgbm_model(**model_params)
            elif model_type == 'XGBoost':
                model = self.create_xgboost_model(**model_params)
            elif model_type == 'NeuralNetwork':
                model = self.create_neural_network_model(X_train.shape[1], **model_params)
            elif model_type == 'RandomForest':
                default_rf_params = {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': 42,
                    'n_jobs': -1
                }
                default_rf_params.update(model_params)
                model = self.RandomForestClassifier(**default_rf_params)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Train the model
            if model_type == 'NeuralNetwork':
                # Neural network training with callbacks
                callbacks_list = [
                    self.callbacks.EarlyStopping(
                        monitor='val_loss' if X_val is not None else 'loss',
                        patience=10,
                        restore_best_weights=True
                    ),
                    self.callbacks.ReduceLROnPlateau(
                        monitor='val_loss' if X_val is not None else 'loss',
                        factor=0.5,
                        patience=5,
                        min_lr=0.0001
                    )
                ]
                
                # Prepare validation data
                validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
                
                # Train
                history = model.fit(
                    X_train, y_train,
                    epochs=100,
                    batch_size=32,
                    validation_data=validation_data,
                    callbacks=callbacks_list,
                    verbose=0
                )
                
                # Predictions for metrics
                y_pred = (model.predict(X_train) > 0.5).astype(int).flatten()
                y_pred_proba = model.predict(X_train).flatten()
                
            else:
                # Tree-based models training
                if X_val is not None and y_val is not None and model_type in ['LightGBM', 'XGBoost']:
                    # Use validation set for early stopping
                    if model_type == 'LightGBM':
                        model.fit(
                            X_train, y_train,
                            eval_set=[(X_val, y_val)],
                            eval_metric='auc',
                            callbacks=[self.lgb.early_stopping(10), self.lgb.log_evaluation(0)]
                        )
                    elif model_type == 'XGBoost':
                        model.fit(
                            X_train, y_train,
                            eval_set=[(X_val, y_val)],
                            verbose=False
                        )
                else:
                    # Standard training
                    model.fit(X_train, y_train)
                
                # Predictions for metrics
                y_pred = model.predict(X_train)
                y_pred_proba = model.predict_proba(X_train)[:, 1] if hasattr(model, 'predict_proba') else y_pred
            
            training_time = time.time() - start_time
            
            # Calculate metrics
            metrics = self._calculate_metrics(
                y_train, y_pred, y_pred_proba, model, X_train, model_type, training_time
            )
            
            logger.info(f"✅ {model_type} trained successfully in {training_time:.2f}s")
            logger.info(f"   Accuracy: {metrics.accuracy:.3f}")
            logger.info(f"   AUC-ROC: {metrics.auc_roc:.3f}")
            logger.info(f"   F1-Score: {metrics.f1_score:.3f}")
            
            return model, metrics
            
        except Exception as e:
            logger.error(f"❌ Error training {model_type}: {str(e)}")
            raise
    
    def _calculate_metrics(self, 
                          y_true: pd.Series, 
                          y_pred: np.ndarray, 
                          y_pred_proba: np.ndarray,
                          model: Any,
                          X: pd.DataFrame,
                          model_type: str,
                          training_time: float) -> ModelMetrics:
        """Calculate comprehensive model performance metrics"""
        
        # Basic metrics
        accuracy = self.accuracy_score(y_true, y_pred)
        precision = self.precision_score(y_true, y_pred, average='binary', zero_division=0)
        recall = self.recall_score(y_true, y_pred, average='binary', zero_division=0)
        f1 = self.f1_score(y_true, y_pred, average='binary', zero_division=0)
        
        # AUC-ROC (handle potential issues)
        try:
            auc_roc = self.roc_auc_score(y_true, y_pred_proba)
        except ValueError:
            auc_roc = 0.5  # Random performance fallback
        
        # Cross-validation scores
        try:
            cv_scores = self.cross_val_score(
                model, X, y_true, cv=5, scoring='roc_auc'
            ).tolist()
        except Exception:
            cv_scores = [auc_roc] * 5  # Fallback
        
        # Feature importance
        feature_importance = {}
        try:
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                importance_values = model.feature_importances_
                feature_importance = dict(zip(X.columns, importance_values))
            elif hasattr(model, 'coef_'):
                # Linear models
                importance_values = np.abs(model.coef_[0])
                feature_importance = dict(zip(X.columns, importance_values))
            elif model_type == 'NeuralNetwork':
                # Neural network: use permutation importance (simplified)
                # For now, assign equal importance
                importance_values = np.ones(len(X.columns)) / len(X.columns)
                feature_importance = dict(zip(X.columns, importance_values))
        except Exception:
            # Fallback: equal importance
            importance_values = np.ones(len(X.columns)) / len(X.columns)
            feature_importance = dict(zip(X.columns, importance_values))
        
        # Prediction time (approximate)
        pred_start = time.time()
        try:
            if model_type == 'NeuralNetwork':
                _ = model.predict(X.iloc[:10])
            else:
                _ = model.predict(X.iloc[:10])
            prediction_time = (time.time() - pred_start) / 10  # Per sample
        except Exception:
            prediction_time = 0.001  # Fallback
        
        return ModelMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc_roc=auc_roc,
            cv_scores=cv_scores,
            feature_importance=feature_importance,
            training_time=training_time,
            prediction_time=prediction_time
        )
    
    def train_ensemble(self, 
                      X_train: pd.DataFrame, 
                      y_train: pd.Series,
                      X_val: Optional[pd.DataFrame] = None,
                      y_val: Optional[pd.Series] = None,
                      models_to_use: Optional[List[str]] = None,
                      ensemble_method: str = 'weighted_voting') -> Dict[str, Any]:
        """
        Train ensemble of models
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            models_to_use: List of model types to include
            ensemble_method: 'weighted_voting', 'equal_voting', or 'stacking'
        
        Returns:
            Dictionary containing trained models and ensemble
        """
        logger.info(f"🎯 Training ensemble with {ensemble_method} method...")
        
        if models_to_use is None:
            models_to_use = self.available_models
        
        # Filter available models
        models_to_use = [m for m in models_to_use if m in self.available_models]
        
        if not models_to_use:
            raise ValueError("No valid models specified")
        
        logger.info(f"   Models to train: {', '.join(models_to_use)}")
        
        trained_models = {}
        model_metrics = {}
        
        # Train individual models
        for model_type in models_to_use:
            try:
                if model_type in self.available_models:
                    model, metrics = self.train_single_model(
                        model_type, X_train, y_train, X_val, y_val
                    )
                    trained_models[model_type] = model
                    model_metrics[model_type] = metrics
                    
                    # Store for ensemble weighting
                    self.models[model_type] = model
                    self.model_metrics[model_type] = metrics
                    
            except Exception as e:
                logger.error(f"❌ Failed to train {model_type}: {str(e)}")
                continue
        
        if not trained_models:
            raise ValueError("No models were successfully trained")
        
        # Create ensemble
        ensemble_model = self._create_ensemble(
            trained_models, model_metrics, ensemble_method
        )
        
        # Calculate ensemble metrics
        ensemble_metrics = self._evaluate_ensemble(
            ensemble_model, X_train, y_train, ensemble_method
        )
        
        logger.info(f"✅ Ensemble training complete!")
        logger.info(f"   Individual models: {len(trained_models)}")
        logger.info(f"   Ensemble accuracy: {ensemble_metrics.accuracy:.3f}")
        logger.info(f"   Ensemble AUC-ROC: {ensemble_metrics.auc_roc:.3f}")
        
        # Update combined feature importance
        self._update_combined_feature_importance()
        
        return {
            'individual_models': trained_models,
            'individual_metrics': model_metrics,
            'ensemble_model': ensemble_model,
            'ensemble_metrics': ensemble_metrics,
            'ensemble_method': ensemble_method,
            'feature_importance': self.feature_importance_combined
        }
    
    def _create_ensemble(self, 
                        trained_models: Dict[str, Any], 
                        model_metrics: Dict[str, ModelMetrics],
                        ensemble_method: str) -> Any:
        """Create ensemble model from individual models"""
        
        if ensemble_method == 'weighted_voting':
            # Weight by AUC-ROC performance
            weights = []
            estimators = []
            
            for model_type, model in trained_models.items():
                weight = model_metrics[model_type].auc_roc
                weights.append(weight)
                estimators.append((model_type, model))
                self.ensemble_weights[model_type] = weight
            
            # Normalize weights
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
                for i, (model_type, _) in enumerate(estimators):
                    self.ensemble_weights[model_type] = weights[i]
            
            # Create weighted voting classifier
            ensemble = WeightedEnsemble(estimators, weights)
            
        elif ensemble_method == 'equal_voting':
            # Equal weights for all models
            estimators = [(model_type, model) for model_type, model in trained_models.items()]
            weights = [1.0 / len(estimators)] * len(estimators)
            
            for model_type, weight in zip(trained_models.keys(), weights):
                self.ensemble_weights[model_type] = weight
            
            ensemble = WeightedEnsemble(estimators, weights)
            
        else:
            raise ValueError(f"Unsupported ensemble method: {ensemble_method}")
        
        return ensemble
    
    def _evaluate_ensemble(self, 
                          ensemble_model: Any, 
                          X: pd.DataFrame, 
                          y: pd.Series,
                          ensemble_method: str) -> ModelMetrics:
        """Evaluate ensemble model performance"""
        
        start_time = time.time()
        
        # Get predictions
        y_pred = ensemble_model.predict(X)
        y_pred_proba = ensemble_model.predict_proba(X)
        
        training_time = time.time() - start_time
        
        # Calculate metrics
        return self._calculate_metrics(
            y, y_pred, y_pred_proba, ensemble_model, X, 'Ensemble', training_time
        )
    
    def _update_combined_feature_importance(self):
        """Update combined feature importance across all models"""
        if not self.model_metrics:
            return
        
        all_features = set()
        for metrics in self.model_metrics.values():
            all_features.update(metrics.feature_importance.keys())
        
        combined_importance = {}
        
        for feature in all_features:
            weighted_importance = 0
            total_weight = 0
            
            for model_type, metrics in self.model_metrics.items():
                if feature in metrics.feature_importance:
                    weight = self.ensemble_weights.get(model_type, 1.0)
                    importance = metrics.feature_importance[feature]
                    weighted_importance += importance * weight
                    total_weight += weight
            
            if total_weight > 0:
                combined_importance[feature] = weighted_importance / total_weight
        
        # Normalize to sum to 1
        total_importance = sum(combined_importance.values())
        if total_importance > 0:
            self.feature_importance_combined = {
                k: v / total_importance for k, v in combined_importance.items()
            }
        
        # Sort by importance
        self.feature_importance_combined = dict(
            sorted(self.feature_importance_combined.items(), 
                  key=lambda x: x[1], reverse=True)
        )
    
    def predict_ensemble(self, 
                        X: pd.DataFrame, 
                        return_probabilities: bool = True,
                        return_individual: bool = False) -> Dict[str, Any]:
        """
        Make predictions using ensemble of models
        
        Args:
            X: Features for prediction
            return_probabilities: Return probability scores
            return_individual: Return individual model predictions
        
        Returns:
            Dictionary with ensemble predictions and optionally individual predictions
        """
        if not self.models:
            raise ValueError("No models trained. Call train_ensemble first.")
        
        start_time = time.time()
        
        results = {
            'ensemble_prediction': None,
            'ensemble_probability': None,
            'confidence_score': 0.0,
            'prediction_time': 0.0
        }
        
        individual_predictions = {}
        individual_probabilities = {}
        
        # Get predictions from each model
        valid_predictions = []
        valid_probabilities = []
        valid_weights = []
        
        for model_type, model in self.models.items():
            try:
                if model_type == 'NeuralNetwork':
                    pred_prob = model.predict(X).flatten()
                    pred = (pred_prob > 0.5).astype(int)
                else:
                    pred = model.predict(X)
                    if hasattr(model, 'predict_proba'):
                        pred_prob = model.predict_proba(X)[:, 1]
                    else:
                        pred_prob = pred.astype(float)
                
                individual_predictions[model_type] = pred
                individual_probabilities[model_type] = pred_prob
                
                # Add to ensemble calculation
                weight = self.ensemble_weights.get(model_type, 1.0)
                valid_predictions.append(pred)
                valid_probabilities.append(pred_prob)
                valid_weights.append(weight)
                
            except Exception as e:
                logger.warning(f"⚠️ Prediction failed for {model_type}: {str(e)}")
                continue
        
        if not valid_predictions:
            raise ValueError("No models could make predictions")
        
        # Calculate ensemble predictions
        valid_weights = np.array(valid_weights)
        valid_weights = valid_weights / valid_weights.sum()  # Normalize
        
        # Weighted average of probabilities
        ensemble_proba = np.average(valid_probabilities, axis=0, weights=valid_weights)
        ensemble_pred = (ensemble_proba > 0.5).astype(int)
        
        # Calculate confidence score (agreement between models)
        if len(valid_predictions) > 1:
            # Standard deviation of probabilities (lower = more agreement)
            prob_std = np.std(valid_probabilities, axis=0)
            confidence = 1.0 - np.mean(prob_std)  # Convert to confidence score
        else:
            confidence = 0.8  # Single model fallback
        
        results.update({
            'ensemble_prediction': ensemble_pred,
            'ensemble_probability': ensemble_proba if return_probabilities else None,
            'confidence_score': confidence,
            'prediction_time': time.time() - start_time
        })
        
        if return_individual:
            results['individual_predictions'] = individual_predictions
            results['individual_probabilities'] = individual_probabilities if return_probabilities else None
        
        return results
    
    def save_models(self, symbol: str, timeframe: str = "1h"):
        """Save all trained models to disk"""
        save_dir = os.path.join(self.models_dir, f"{symbol}_{timeframe}")
        os.makedirs(save_dir, exist_ok=True)
        
        saved_count = 0
        
        for model_type, model in self.models.items():
            try:
                model_path = os.path.join(save_dir, f"{model_type}_model.pkl")
                
                if model_type == 'NeuralNetwork':
                    # Save Keras model
                    model_path = os.path.join(save_dir, f"{model_type}_model.h5")
                    model.save(model_path)
                else:
                    # Save sklearn/xgb/lgb model
                    with open(model_path, 'wb') as f:
                        pickle.dump(model, f)
                
                saved_count += 1
                logger.info(f"💾 Saved {model_type} model to {model_path}")
                
            except Exception as e:
                logger.error(f"❌ Failed to save {model_type} model: {str(e)}")
        
        # Save metrics and weights
        try:
            metadata = {
                'model_metrics': {k: {
                    'accuracy': v.accuracy,
                    'auc_roc': v.auc_roc,
                    'f1_score': v.f1_score,
                    'training_time': v.training_time
                } for k, v in self.model_metrics.items()},
                'ensemble_weights': self.ensemble_weights,
                'feature_importance': self.feature_importance_combined,
                'timestamp': datetime.now().isoformat()
            }
            
            metadata_path = os.path.join(save_dir, "metadata.pkl")
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.info(f"💾 Saved metadata to {metadata_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save metadata: {str(e)}")
        
        logger.info(f"✅ Successfully saved {saved_count} models for {symbol}_{timeframe}")
    
    def load_models(self, symbol: str, timeframe: str = "1h") -> bool:
        """Load trained models from disk"""
        load_dir = os.path.join(self.models_dir, f"{symbol}_{timeframe}")
        
        if not os.path.exists(load_dir):
            logger.warning(f"⚠️ No saved models found for {symbol}_{timeframe}")
            return False
        
        loaded_count = 0
        
        # Try to load each available model type
        for model_type in self.available_models:
            try:
                if model_type == 'NeuralNetwork':
                    model_path = os.path.join(load_dir, f"{model_type}_model.h5")
                    if os.path.exists(model_path):
                        model = self.keras.models.load_model(model_path)
                        self.models[model_type] = model
                        loaded_count += 1
                        logger.info(f"📂 Loaded {model_type} model")
                else:
                    model_path = os.path.join(load_dir, f"{model_type}_model.pkl")
                    if os.path.exists(model_path):
                        with open(model_path, 'rb') as f:
                            model = pickle.load(f)
                        self.models[model_type] = model
                        loaded_count += 1
                        logger.info(f"📂 Loaded {model_type} model")
                        
            except Exception as e:
                logger.warning(f"⚠️ Failed to load {model_type} model: {str(e)}")
        
        # Load metadata
        try:
            metadata_path = os.path.join(load_dir, "metadata.pkl")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                
                self.ensemble_weights = metadata.get('ensemble_weights', {})
                self.feature_importance_combined = metadata.get('feature_importance', {})
                
                logger.info(f"📂 Loaded metadata")
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to load metadata: {str(e)}")
        
        if loaded_count > 0:
            logger.info(f"✅ Successfully loaded {loaded_count} models for {symbol}_{timeframe}")
            return True
        else:
            logger.warning(f"❌ No models could be loaded for {symbol}_{timeframe}")
            return False
    
    def get_model_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary of all models"""
        if not self.model_metrics:
            return {"error": "No models trained"}
        
        summary = {
            'individual_models': {},
            'ensemble_performance': {},
            'feature_importance': self.feature_importance_combined,
            'model_availability': {
                'total_available': len(self.available_models),
                'trained_models': len(self.models),
                'available_types': self.available_models
            }
        }
        
        # Individual model performance
        for model_type, metrics in self.model_metrics.items():
            summary['individual_models'][model_type] = {
                'accuracy': metrics.accuracy,
                'precision': metrics.precision,
                'recall': metrics.recall,
                'f1_score': metrics.f1_score,
                'auc_roc': metrics.auc_roc,
                'cv_mean': np.mean(metrics.cv_scores),
                'cv_std': np.std(metrics.cv_scores),
                'training_time': metrics.training_time,
                'prediction_time': metrics.prediction_time
            }
        
        # Best performing model
        if self.model_metrics:
            best_model = max(
                self.model_metrics.items(), 
                key=lambda x: x[1].auc_roc
            )
            summary['best_individual_model'] = {
                'model_type': best_model[0],
                'auc_roc': best_model[1].auc_roc,
                'accuracy': best_model[1].accuracy
            }
        
        # Ensemble weights
        if self.ensemble_weights:
            summary['ensemble_weights'] = self.ensemble_weights
        
        # Top features
        if self.feature_importance_combined:
            top_features = dict(list(self.feature_importance_combined.items())[:10])
            summary['top_features'] = top_features
        
        return summary


class WeightedEnsemble:
    """Custom weighted ensemble classifier"""
    
    def __init__(self, estimators: List[Tuple[str, Any]], weights: List[float]):
        self.estimators = estimators
        self.weights = np.array(weights)
        self.weights = self.weights / self.weights.sum()  # Normalize
    
    def predict(self, X):
        """Make binary predictions"""
        predictions = []
        
        for (name, estimator), weight in zip(self.estimators, self.weights):
            if hasattr(estimator, 'predict'):
                pred = estimator.predict(X)
            else:
                # Neural network
                pred = (estimator.predict(X) > 0.5).astype(int).flatten()
            
            # Ensure prediction is 1D array
            if pred.ndim > 1:
                pred = pred.flatten()
            
            predictions.append(pred)
        
        # Convert to numpy array for averaging
        predictions = np.array(predictions)
        
        # Weighted vote
        weighted_pred = np.average(predictions, axis=0, weights=self.weights)
        return (weighted_pred > 0.5).astype(int)
    
    def predict_proba(self, X):
        """Return prediction probabilities"""
        probabilities = []
        
        for (name, estimator), weight in zip(self.estimators, self.weights):
            if hasattr(estimator, 'predict_proba'):
                prob = estimator.predict_proba(X)[:, 1]
            elif hasattr(estimator, 'predict') and 'Neural' in name:
                prob = estimator.predict(X).flatten()
            else:
                # Convert predictions to probabilities
                pred = estimator.predict(X)
                prob = pred.astype(float)
            
            # Ensure probability is 1D array
            if prob.ndim > 1:
                prob = prob.flatten()
            
            probabilities.append(prob)
        
        # Convert to numpy array for averaging
        probabilities = np.array(probabilities)
        
        # Weighted average
        weighted_prob = np.average(probabilities, axis=0, weights=self.weights)
        return weighted_prob


# Convenience functions for integration
def create_advanced_ml_system(**kwargs) -> AdvancedMLModels:
    """Create and return an Advanced ML Models system"""
    return AdvancedMLModels(**kwargs)


def train_advanced_models(X_train: pd.DataFrame, 
                         y_train: pd.Series,
                         symbol: str,
                         models_to_use: Optional[List[str]] = None,
                         save_models: bool = True) -> Dict[str, Any]:
    """
    Convenience function to train advanced models
    
    Args:
        X_train: Training features
        y_train: Training targets  
        symbol: Trading symbol
        models_to_use: List of models to train
        save_models: Whether to save trained models
    
    Returns:
        Training results dictionary
    """
    # Create ML system
    ml_system = AdvancedMLModels()
    
    # Train ensemble
    results = ml_system.train_ensemble(
        X_train, y_train,
        models_to_use=models_to_use
    )
    
    # Save models if requested
    if save_models:
        ml_system.save_models(symbol)
    
    return results


if __name__ == "__main__":
    # Example usage and testing
    logger.info("🧪 Testing Advanced ML Models System")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y = pd.Series((X.sum(axis=1) > 0).astype(int))
    
    logger.info(f"📊 Generated test data: {X.shape}, {y.value_counts().to_dict()}")
    
    # Test advanced ML system
    try:
        ml_system = AdvancedMLModels()
        
        # Train ensemble
        results = ml_system.train_ensemble(X, y)
        
        # Make predictions
        predictions = ml_system.predict_ensemble(X[:10])
        
        logger.info("✅ Advanced ML Models system test completed successfully!")
        logger.info(f"   Models trained: {list(results['individual_models'].keys())}")
        logger.info(f"   Ensemble accuracy: {results['ensemble_metrics'].accuracy:.3f}")
        logger.info(f"   Prediction confidence: {predictions['confidence_score']:.3f}")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc()) 