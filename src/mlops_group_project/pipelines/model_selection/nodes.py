import logging
from typing import Tuple, Union
import warnings; warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.base import BaseEstimator

import optuna
import mlflow
from mlflow.tracking import MlflowClient


log = logging.getLogger(__name__)


# NOTE: to avoid changing the champion model randomly during dev runs (champion model is our prod model)
# our approach is to store promosiing models as "challengers". Which after further testing (e.g. shadow testing)
# can indeed be manually promoted to champion. This is a common practice in ML engineering.


# TODO add random state


# should this also go into a 'utils'-like module?
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


# NOTE: if true champion model is stored locally then this probably won't be necessary
def load_registered_model_version(model_name: str, version: int = -1) -> Union[BaseEstimator, None]:
    """
    Load a specific version of a registered model from the
    MLflow Model Registry. If version is -1, load the latest version.
    Useful: https://mlflow.org/docs/latest/model-registry.html    
    
    Args:
        model_name (str): The name of the registered model.
        version (int): The version of the model to load, or -1 for the latest version.
    
    Returns:
        model: The loaded model. Or None if there is no matching model.
    """
    client = MlflowClient()

    if version == -1:
        # get the latest version
        versions = client.get_latest_versions(model_name)
        if not versions:
            raise Exception(f"No versions found for model {model_name}")
        latest_version = max(versions, key=lambda v: v.version).version
        version = latest_version
    
    model_uri = f"models:/{model_name}/{version}"
    try:
        model = mlflow.sklearn.load_model(model_uri)
        return model
    except Exception as e:
        raise Exception(f"Could not load model from Registry: {e}")
    
    
def locate_champion_model(alias: str = "champion") -> Union[BaseEstimator, None]:
    """
    Locate the champion model in the MLflow Model Registry.
    
    Returns:
        model: The champion model. Or None if there is no champion model.
    """
    client = MlflowClient()
    
    models = client.search_registered_models()
    for model in models:
        for alias in model.aliases.keys():
            if alias == "champion":
                return load_registered_model_version(model.name, model.aliases[alias])
    
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
    n_trials: int = 50,
    models: dict = MODELS_DICT
) -> BaseEstimator:
    
    X_combined = pd.concat([X_train, X_val], axis=0)
    y_combined = pd.concat([y_train, y_val], axis=0)
        
    best_model = None
    best_score = 0
    best_run_id = None
    best_model_name = None
    
    mlflow.autolog(log_model_signatures=True, log_input_examples=True)
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
                
                model.fit(X_combined, np.ravel(y_combined))
                train_pred = model.predict(X_combined)
                test_pred = model.predict(X_test)
                
                train_report = classification_report(np.ravel(y_combined), train_pred, output_dict=True)
                test_report = classification_report(np.ravel(y_test), test_pred, output_dict=True)
                
                overfitting = np.abs(train_report["1"]["f1-score"] - test_report["1"]["f1-score"])
                
                # updating best model
                if (overfitting < .1) & (test_report["1"]["f1-score"] > best_score):
                    best_model = model
                    best_score = test_report["1"]["f1-score"]
                    best_run_id = run.info.run_id
                    best_model_name = model_name

                # NOTE: not all items are logged, so we are adding more
                # metrics logging (train and test)
                mlflow.log_metric("training_precision_1", train_report["1"]["precision"])
                mlflow.log_metric("training_recall_1", train_report["1"]["recall"])
                mlflow.log_metric("training_f1_score_1", train_report["1"]["f1-score"])
                mlflow.log_metric("training_accuracy", train_report["accuracy"])
                
                mlflow.log_metric("testing_precision_1", test_report["1"]["precision"])
                mlflow.log_metric("testing_recall_1", test_report["1"]["recall"])
                mlflow.log_metric("testing_f1_score_1", test_report["1"]["f1-score"])
                mlflow.log_metric("testing_accuracy", test_report["accuracy"])
                
                log.info(f"Model: {model_name}")
                log.info(f"Training F1-score: {train_report['1']['f1-score']}")
                log.info(f"Testing F1-score: {test_report['1']['f1-score']}")
                log.info(f"Overfitting: {overfitting}")
        
        # if models show great overfitting then we don't register any model
        if best_model is not None:
            
            log.info(f"Best model: {best_model_name}")
                       
            # loading the champion model
            champion_model = locate_champion_model()
            
            # comparing challenger to champion
            if champion_model is not None:
                # check this out: https://mlflow.org/docs/latest/model-registry.html
                champion_test_pred = champion_model.predict(X_test)
                champion_f1 = f1_score(np.ravel(y_test), champion_test_pred, pos_label=1)
                
                print(f"Champion model F1-score: {champion_f1}")
                
                if best_score >= champion_f1:
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


if __name__ == '__main__':
    # testing
    import os

    params = {
        'X_train': pd.read_csv(os.path.join('data', 'dev', '07_model_input', 'X_train_selected.csv')),
        'y_train': pd.read_csv(os.path.join('data', 'dev', '05_split', 'y_train.csv')),
        'X_val': pd.read_csv(os.path.join('data', 'dev', '07_model_input', 'X_val_selected.csv')),
        'y_val': pd.read_csv(os.path.join('data', 'dev', '05_split', 'y_val.csv')),
        'X_test': pd.read_csv(os.path.join('data', 'dev', '07_model_input', 'X_test_selected.csv')),
        'y_test': pd.read_csv(os.path.join('data', 'dev', '05_split', 'y_test.csv')),
        'n_trials': 1,
        'models': MODELS_DICT
    }
    
    select_model(**params)