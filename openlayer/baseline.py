from Automunge import AutoMunge
import pandas as pd
import numpy as np
import autosklearn.classification

import autosklearn.estimators
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder

from ConfigSpace.configuration_space import ConfigurationSpace
from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import SPARSE, DENSE, UNSIGNED_DATA, INPUT


class NoPreprocessing(AutoSklearnPreprocessingAlgorithm):
    def __init__(self, **kwargs):
        """ This preprocessors does not change the data """
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


class QuickBaseline:
    def __init__(
        self,
        train_df: pd.DataFrame = None,
        val_df: pd.DataFrame = None,
        df: pd.DataFrame = None,
        max_training_size: int = 200000,
        max_validation_size: int = 10000,
        ensemble_size: int = 10,
        label_column: str = "label",
        random_seed: int = 0,
    ):
        if train_df is None:
            if df is None:
                raise Exception("No dataframe found.")
            train_df, val_df = self.split_dataset(
                df, random_seed, max_training_size, max_validation_size
            )

        if val_df is None:
            raise Exception("No validation dataframe found.")

        self.train_df = train_df.dropna()
        self.val_df = val_df.dropna()
        self.label_column = label_column
        self.column_names = self.train_df.drop(
            self.label_column, axis=1
        ).columns.tolist()
        self.ensemble_size = ensemble_size

        self.category_names = None
        self.process_dict = None
        self.train_features = None
        self.train_labels = None

    def process_dataset(self,):
        # Run processor
        processor = AutoMunge()
        # print('train_df', self.train_df)
        args = processor.automunge(
            self.train_df,
            labels_column=self.label_column,
            NArw_marker=False,
            printstatus=False,
        )
        process_dict = args[-1]

        # get features
        train_features, train_ids, train_labels, _ = processor.postmunge(
            process_dict, self.train_df, printstatus=False
        )

        cols = args[0].columns.to_list()
        cat_names = []
        transformation_list = ["_1010", "_bnry", "_hsh2", "_hash"]

        for col in cols:
            ending = col[-5:]

            if ending != "_nmbr":
                for transformation in transformation_list:
                    split_name = col.split(transformation)
                    if len(split_name) > 1:
                        cat_names.append(split_name[0])
                        break

        self.process_dict = process_dict
        self.train_features = train_features
        self.train_labels = self.train_df[self.label_column]
        self.category_names = list(set(cat_names))

    @staticmethod
    def split_dataset(
        df: pd.DataFrame, random_seed, max_training_size, max_validation_size
    ):
        size = len(df)

        if size < 30000:
            max_training_size, max_validation_size = (
                int(0.7 * size),
                (size - int(0.7 * size)),
            )

        df = df.sample(random_state=random_seed)
        train_df = df[:max_training_size]
        val_df = df[max_training_size : max_training_size + max_validation_size]

        return train_df, val_df

    def get_auto(self, timeout: int, per_run_limit: int):
        model = autosklearn.estimators.AutoSklearnClassifier(
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
        )

        model.fit(self.train_features, self.train_labels, dataset_name="auto")

        return model

    def transform_to_voter(self, auto_clf, y):
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
    def set_ensemble(models, weights, y):

        voter = VotingClassifier(estimators=None, voting="soft")

        voter.estimators = models
        voter.estimators_ = models
        voter.weights = weights
        voter.le_ = LabelEncoder().fit(y)
        voter.classes_ = voter.le_.classes_

        return voter

    @staticmethod
    def to_sklearn(steps):
        model = None

        for name, step in steps:
            if name == "classifier":
                model = step.choice.estimator

        return model

    def get_model_and_function(self, timeout: int, per_run_limit: int):
        def predict_proba(
            model, input_features, col_names, processor, process_dict,
        ):
            # from Automunge import AutoMunge
            df = pd.DataFrame(input_features, columns=col_names)
            features, _ids, _labels, _ = processor.postmunge(
                process_dict, df, printstatus=False
            )

            input_feats = features.to_numpy().astype(float)
            predictions = np.average(
                model._collect_probas(input_feats), axis=0, weights=model.weights
            )

            return predictions

        auto_clf = self.get_auto(timeout, per_run_limit)
        voter_clf = self.transform_to_voter(
            auto_clf, self.train_df.copy()[self.label_column]
        )

        return (
            voter_clf,
            predict_proba,
            self.column_names,
            self.category_names,
            self.process_dict,
        )
