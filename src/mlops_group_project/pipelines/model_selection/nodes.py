import logging
from typing import Tuple, Union, Dict
import warnings; warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc)
from sklearn.base import BaseEstimator

import optuna

import mlflow
from mlflow.tracking import MlflowClient


log = logging.getLogger(__name__)


# NOTE: to avoid changing the champion model randomly during dev runs (champion model is our prod model)
# our approach is to store promising models as "challengers". Which after further testing (e.g. shadow testing)
# can indeed be manually promoted to champion. This is a common practice in ML engineering.

MODELS_DICT = {
    'RandomForestClassifier': {
        'model': RandomForestClassifier(),
        'tune': True
        },
    
    'GradientBoostingClassifier': {
        'model': GradientBoostingClassifier(),
        'tune': True
    },
    
    'LogisticRegression': {
        'model': LogisticRegression(),
        'tune': True
    },
}


def register_model(
    model_path: str,
    model_name: str,
    model_tag: str = None,
    model_alias: str = None
) -> None:
    """
    Register a model in the MLflow Model Registry.
    
    Args:
        model_path (str): The path to the model to register.
        model_name (str): The name of the model to register.
        model_tag (str): A tag to add to the model.
        model_version (int): The version of the model to register.
        model_alias (str): An alias to add to the model.
    """
    
    client = MlflowClient()
    
    # registering model
    version = mlflow.register_model(
        model_path, model_name
    )
    
    if any(var is not None for var in [model_tag, model_alias]):
    
        if model_tag:
            client.set_registered_model_tag(
                model_name, "task", "classification"
            )
        
        # creating/reassigning alias
        if model_alias:
            client.set_registered_model_alias(
                model_name, model_alias, version.version
            )


def locate_champion_model_metrics() -> Union[Dict[str, float], None]:
        """
        Locate the champion model in the MLflow Model Registry and returns its metrics.
        
        Returns:
            model: The champion model. Or None if there is no champion model.
        """
        client = MlflowClient()
        models = client.search_registered_models()
        
        for model in models:
            for alias in model.aliases.keys():
                if alias == "champion":
                    version = int(model.aliases[alias]) - 1  # e.g. version '1' in the list is actually the index 0 (it starts at 1)
                    metrics = client.get_run(model.latest_versions[version].run_id).data.metrics
                    return metrics
        
        return None


def optuna_objective(
    trial: optuna.Trial,
    train: Tuple[pd.DataFrame, pd.Series],
    val: Tuple[pd.DataFrame, pd.Series],
    model_name: str
) -> float:
    
    model = MODELS_DICT[model_name]['model']
    
    X_train, y_train = train
    X_val, y_val = val
    
    # unfortunately there really is not better way than if-else statements
    if model_name == 'RandomForestClassifier':
        n_estimators = trial.suggest_int('n_estimators', 10, 100)
        max_depth = trial.suggest_categorical('max_depth', [None, 5, 10])
        class_weight = trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None])
        
        model.set_params(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight
        )
    
    elif model_name == 'GradientBoostingClassifier':
        n_estimators = trial.suggest_int('n_estimators', 10, 100)
        learning_rate = trial.suggest_loguniform('learning_rate', 0.01, 1)
        max_depth = trial.suggest_categorical('max_depth', [3, 5, 10])
        
        model.set_params(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth
        )
        
    elif model_name == 'LogisticRegression':
        penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])
        C = trial.suggest_loguniform('C', 0.01, 10)
        solver = 'liblinear' if penalty == 'l1' else 'lbfgs'
        
        model.set_params(
            penalty=penalty,
            C=C,
            solver=solver
        )
    
    else:
        raise ValueError(f"Model {model_name} not found in MODELS_DICT")
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    score = f1_score(y_val, y_pred, pos_label=1) # f1-score for the pos. class (i.e. readmitted)
    
    return score


def select_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_trials: int = 20,
    models: dict = MODELS_DICT
) -> BaseEstimator:
    
    X_combined = pd.concat([X_train, X_val], axis=0)
    y_combined = pd.concat([y_train, y_val], axis=0)
        
    best_model = None
    best_score = 0
    best_run_id = None
    best_model_name = None
    
    with mlflow.start_run(run_name="model_selection_run", nested=True):
        for model_name, model_info in models.items():
            with mlflow.start_run(run_name=model_name, nested=True) as run:
                
                print(model_name, run.info.run_id)
                
                if model_info['tune']:
                    study = optuna.create_study(direction='maximize')
                    study.optimize(
                        lambda trial: optuna_objective(
                            trial,
                            (X_train, np.ravel(y_train)),
                            (X_val, np.ravel(y_val)),
                            model_name
                        ),
                        n_trials=n_trials
                    )
                    model = model_info['model'].set_params(**study.best_params)
                else:
                    model = model_info['model']
                
                mlflow.autolog()
                
                model.fit(X_combined, np.ravel(y_combined))
                train_pred = model.predict(X_combined)
                test_pred = model.predict(X_test)
                test_pred_proba = model.predict_proba(X_test)
                
                train_report = classification_report(np.ravel(y_combined), train_pred, output_dict=True)
                test_report = classification_report(np.ravel(y_test), test_pred, output_dict=True)
                
                overfitting = np.abs(train_report["1"]["f1-score"] - test_report["1"]["f1-score"])
                
                # updating best model
                if (overfitting < .1) & (test_report["1"]["f1-score"] > best_score):
                    best_model = model
                    best_score = test_report["1"]["f1-score"]
                    best_run_id = run.info.run_id
                    best_model_name = model_name
                    
                mlflow.log_metric("training_precision_1", train_report["1"]["precision"])
                mlflow.log_metric("training_recall_1", train_report["1"]["recall"])
                mlflow.log_metric("training_f1_score_1", train_report["1"]["f1-score"])
                mlflow.log_metric("training_accuracy", train_report["accuracy"])
                
                mlflow.log_metric("testing_precision_1", test_report["1"]["precision"])
                mlflow.log_metric("testing_recall_1", test_report["1"]["recall"])
                mlflow.log_metric("testing_f1_score_1", test_report["1"]["f1-score"])
                mlflow.log_metric("testing_accuracy", test_report["accuracy"])
                
                mlflow.log_metric("overfitting", overfitting)
                
                # Compute and log normalized confusion matrix
                cm = confusion_matrix(y_test, test_pred, normalize='true')
                plt.figure(figsize=(10,7))
                sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues')
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title('Normalized Confusion Matrix')
                mlflow.log_figure(plt.gcf(), "normalized_confusion_matrix.png")
                plt.close()
                
                # roc curve
                plt.figure()
                fpr, tpr, _ = roc_curve(y_test, test_pred_proba[:, 1])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curve (Positive Class)')
                plt.legend(loc='lower right')
                mlflow.log_figure(plt.gcf(), "roc_curve_positive_class.png")
                plt.close()
                
                log.info(f"Model: {model_name}")
                log.info(f"Training F1-score: {train_report['1']['f1-score']}")
                log.info(f"Testing F1-score: {test_report['1']['f1-score']}")
                log.info(f"Overfitting: {overfitting}")
        
        # if models show great overfitting then we don't register any model
        if best_model is not None:
            
            log.info(f"Best model: {best_model_name}")
                       
            # loading the champion model score
            champion_model_metrics = locate_champion_model_metrics()
            
            # comparing challenger to champion
            if champion_model_metrics is not None:
                champion_model_score = champion_model_metrics['testing_f1_score_1']
                print(f"Champion model F1-score: {champion_model_score}")
                
                # a promising model, i.e. challenger, is a model whose score is within 0.1 of the
                # champion model (better or worse, we allow this flexibility since the models
                # were tested in possibly different datasets so their scores are not directly comparable)
                score_diff = abs(best_score - champion_model_score)
                log.info(f"Score difference: {score_diff}")
                
                if score_diff < 0.1:
                    
                    log.info("Challenger model might be better than champion model.")
                    
                    # register the best model as the a promising challenger
                    register_model(
                        f"runs:/{best_run_id}/model",
                        best_model_name,
                        model_alias="challenger"
                    )
            else:
                # if there is no champion model we register the best model as the champion
                # (this will only be triggered in the 1st run ever - long before an actual model is put into production)
                register_model(
                    f"runs:/{best_run_id}/model",
                    best_model_name,
                    model_alias="champion"
                )
         
    return best_model
    
    