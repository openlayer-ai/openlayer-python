from typing import Callable, Dict, List, Tuple

import autosklearn.classification
import autosklearn.estimators
import numpy as np
import pandas as pd
from Automunge import AutoMunge
from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import DENSE, INPUT, SPARSE, UNSIGNED_DATA
from ConfigSpace.configuration_space import ConfigurationSpace
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder

# ------------------------------- MONKEY PATCH ------------------------------- #
"""Include the option of not preprocessing to auto-sklearn."""


class NoPreprocessing(AutoSklearnPreprocessingAlgorithm):
    def __init__(self, **kwargs):
        """This preprocessors does not change the data"""
        # Some internal checks makes sure parameters are set
        for key, val in kwargs.items():
            setattr(self, key, val)

    def fit(self, X, Y=None):
        return self

    def transform(self, X):
        return X

    @staticmethod
    def get_properties(dataset_properties=None):
        return {
            "shortname": "NoPreprocessing",
            "name": "NoPreprocessing",
            "handles_regression": True,
            "handles_classification": True,
            "handles_multiclass": True,
            "handles_multilabel": True,
            "handles_multioutput": True,
            "is_deterministic": True,
            "input": (SPARSE, DENSE, UNSIGNED_DATA),
            "output": (INPUT,),
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        return ConfigurationSpace()  # Return an empty configuration as there is None


# Add NoPreprocessing component to auto-sklearn.
autosklearn.pipeline.components.data_preprocessing.add_preprocessor(NoPreprocessing)

# ----------------------------- END MONKEY PATCH ----------------------------- #


class QuickBaseline:
    def __init__(
        self,
        label_column_name: str,
        train_df: pd.DataFrame = None,
        ensemble_size: int = 10,
        random_seed: int = 0,
    ):
        self.train_df = train_df.dropna()
        self.label_column_name = label_column_name
        self.column_names = self.train_df.drop(
            self.label_column_name, axis=1
        ).columns.tolist()
        self.train_labels = self.train_df[self.label_column_name]
        self.ensemble_size = ensemble_size
        self.random_seed = random_seed

    def preprocess_dataset(self) -> Tuple[Dict[any, any], pd.DataFrame]:
        """Preprocesses the training set prior to fitting the model.
        Processing steps include normalizing, encoding, and others.

        Returns:
            Tuple[Dict[any, any], pd.DataFrame]: tuple with preprocessing dict
                and transformed dataset
        """
        # Run AutoMunge to find the suggested feature preprocessing
        preprocessor = AutoMunge()
        args = preprocessor.automunge(
            self.train_df,
            labels_column=self.label_column_name,
            NArw_marker=False,
            printstatus=False,
        )

        # Get the features metadata, including the normalization and encoding types
        preprocessing_dict = args[-1]

        # Apply the transformations to the training set
        train_features_df, _, _, _ = preprocessor.postmunge(
            preprocessing_dict, self.train_df, printstatus=False
        )

        return (preprocessing_dict, train_features_df)

    def get_categorical_feature_names(
        self, train_features_df: pd.DataFrame
    ) -> List[str]:
        """Get the list of categorical feature names based on the preprocessed
        dataset

        Args:
            train_features_df (pd.DataFrame): pandas df with the processed training set

        Returns:
           List[str]: list with the categorical feature names
        """

        column_names = train_features_df.columns.to_list()

        categorical_feature_names = []
        categorical_transformation_list = ["_1010", "_bnry", "_hsh2", "_hash"]

        for col in column_names:
            # last 5 characters indicate the type of transformation applied
            ending = col[-5:]

            if ending != "_nmbr":
                for transformation in categorical_transformation_list:
                    split_name = col.split(transformation)
                    if len(split_name) > 1:
                        categorical_feature_names.append(split_name[0])
                        break

        return list(set(categorical_feature_names))

    def train_auto_classifiers(
        self, timeout: int, per_run_limit: int, train_features_df: pd.DataFrame
    ):
        """Automatically finds and trains the classifiers suited for task

        Args:
            timeout (int): time limit in seconds for the search of appropriate models
            per_run_limit (int): time limit for a single call to the machine learning
                model. Model fitting will be terminated if the ML algorithm runs over
                the time limit
            train_features_df (pd.DataFrame): processed training set

        Returns:
            model: trained model object
        """
        auto_clf = autosklearn.estimators.AutoSklearnClassifier(
            time_left_for_this_task=timeout,
            per_run_time_limit=per_run_limit,
            ensemble_size=self.ensemble_size,
            initial_configurations_via_metalearning=0,
            include={
                "data_preprocessor": ["NoPreprocessing"],
                "feature_preprocessor": ["no_preprocessing"],
                "classifier": [
                    "adaboost",
                    "bernoulli_nb",
                    "decision_tree",
                    "extra_trees",
                    "gaussian_nb",
                    "gradient_boosting",
                    "mlp",
                    "multinomial_nb",
                    "random_forest",
                    "sgd",
                ],
            },
            seed=self.random_seed,
        )

        # Fit classifiers
        auto_clf.fit(train_features_df, self.train_labels, dataset_name="auto")

        # Combine classifiers using a voting scheme
        voter_clf = self.transform_to_voter(
            auto_clf, self.train_df.copy()[self.label_column_name]
        )

        return voter_clf

    def transform_to_voter(self, auto_clf, y: pd.DataFrame):
        """Combines AutoSklearn models in a voting scheme and
        transforms them to Sklearn models

        Args:
            auto_clf (AutoSklearnClassifier object): AutoSklearn model
            y (pd.DataFrame): dataset labels

        Returns:
            sklearn.ensemble.VotingClassifier object: voting classifier
        """
        _weights = auto_clf.automl_.ensemble_.weights_
        _id = auto_clf.automl_.ensemble_.identifiers_
        _models = auto_clf.automl_.models_

        models = []
        weights = []

        for weight, identifier in zip(list(_weights), list(_id)):
            if weight == 0.0:
                continue

            weights.append(weight)
            sklearn_model = self.to_sklearn(_models[identifier].steps)
            models.append(sklearn_model)

        return self.set_ensemble(models, weights, y)

    @staticmethod
    def set_ensemble(models: List[object], weights: List[float], y: pd.DataFrame):
        """Sets corresponding attributes for the Voting classifier

        Args:
            models (List[object]): list of models to be combined by voting
            weights (List[float]): list of weights for the voting
            y (pd.DataFrame): dataset labels

        Returns:
            sklearn.ensemble.VotingClassifier object: voting classifier
        """
        voter = VotingClassifier(estimators=None, voting="soft")

        voter.estimators = models
        voter.estimators_ = models
        voter.weights = weights
        voter.le_ = LabelEncoder().fit(y)
        voter.classes_ = voter.le_.classes_

        return voter

    @staticmethod
    def to_sklearn(steps: Tuple[str, object]):
        """Converts the auto-sklearn model objects to sklearn for
        model deployment purposes.

        Args:
            steps (Tuple[str, object]): tuple with name and pipeline objects

        Returns:
            model: model as an sklearn model object
        """
        model = None

        for name, step in steps:
            if name == "classifier":
                model = step.choice.estimator

        return model

    def get_predict_function(self) -> Callable:
        """Defines the predict function for the trained model"""

        def predict_proba(
            model,
            input_features,
            col_names,
            preprocessor,
            preprocessing_dict,
        ):
            df = pd.DataFrame(input_features, columns=col_names)
            features, _, _, _ = preprocessor.postmunge(
                preprocessing_dict, df, printstatus=False
            )

            input_feats = features.to_numpy(dtype="O")
            predictions = np.average(
                model._collect_probas(input_feats), axis=0, weights=model.weights
            )

            return predictions

        return predict_proba
