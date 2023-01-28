"""Schemas for the objects that shall be created on the Openlayer platform.
"""
import marshmallow as ma

from .datasets import DatasetType
from .models import ModelType
from .tasks import TaskType


class CommitSchema(ma.Schema):
    """Schema for commits."""

    commitMessage = ma.fields.Str(
        required=True,
        validate=ma.validate.Length(
            min=1,
            max=140,
        ),
    )


class DatasetSchema(ma.Schema):
    """Schema for datasets."""

    categoricalFeatureNames = ma.fields.List(
        ma.fields.Str(), allow_none=True, load_default=[]
    )
    classNames = ma.fields.List(ma.fields.Str(), required=True)
    columnNames = ma.fields.List(
        ma.fields.Str(),
        required=True,
    )
    label = ma.fields.Str(
        validate=ma.validate.OneOf(
            [dataset_type.value for dataset_type in DatasetType],
            error="`label` must be one of the supported frameworks."
            + "Check out our API reference for a full list"
            + " https://reference.openlayer.com/reference/api/openlayer.DatasetType.html.\n ",
        ),
        required=True,
    )
    featureNames = ma.fields.List(
        ma.fields.Str(),
        load_default=[],
    )
    labelColumnName = ma.fields.Str(required=True)
    language = ma.fields.Str(
        load_default="en",
        validate=ma.validate.Regexp(
            r"^[a-z]{2}(-[A-Z]{2})?$",
            error="`language` of the dataset is not in the ISO 639-1 (alpha-2 code) format.",
        ),
    )
    metadata = ma.fields.Dict(allow_none=True, load_default={})
    predictionsColumnName = ma.fields.Str(allow_none=True, load_default=None)
    sep = ma.fields.Str(load_default=",")
    textColumnName = ma.fields.Str(
        allow_none=True,
    )

    @ma.validates_schema
    def validates_label_column_not_in_feature_names(self, data, **kwargs):
        """Validates whether the label column name is not on the feature names list"""
        if data["labelColumnName"] in data["featureNames"]:
            raise ma.ValidationError(
                f"`labelColumnName` `{data['labelColumnName']}` must not be in `featureNames`."
            )


class ModelSchema(ma.Schema):
    """Schema for models with artifacts (i.e., model_package)."""

    categoricalFeatureNames = ma.fields.List(ma.fields.Str(), load_default=[])
    classNames = ma.fields.List(
        ma.fields.Str(),
    )
    name = ma.fields.Str(
        required=True,
        validate=ma.validate.Length(
            min=1,
            max=64,
        ),
    )
    featureNames = ma.fields.List(ma.fields.Str(), allow_none=True, load_default=[])
    metadata = ma.fields.Dict(
        allow_none=True,
        load_default={},
    )
    architectureType = ma.fields.Str(
        validate=ma.validate.OneOf(
            [model_framework.value for model_framework in ModelType],
            error="`architectureType` must be one of the supported frameworks."
            + " Check out our API reference for a full list"
            + " https://reference.openlayer.com/reference/api/openlayer.ModelType.html.\n ",
        ),
        required=True,
    )


class ProjectSchema(ma.Schema):
    """Schema for projects."""

    description = ma.fields.Str(
        validate=ma.validate.Length(
            min=1,
            max=140,
        ),
    )
    name = ma.fields.Str(
        required=True,
        validate=ma.validate.Length(
            min=1,
            max=64,
        ),
    )
    task_type = ma.fields.Str(
        alidate=ma.validate.OneOf(
            [task_type.value for task_type in TaskType],
            error="`task_type` must be one of the supported tasks."
            + " Check out our API reference for a full list"
            + " https://reference.openlayer.com/reference/api/openlayer.TaskType.html.\n ",
        ),
    )


class ShellModelSchema(ma.Schema):
    """Schema for models without artifacts (i.e., shell model)."""

    name = ma.fields.Str(
        required=True,
        validate=ma.validate.Length(
            min=1,
            max=64,
        ),
    )
    metadata = ma.fields.Dict(
        allow_none=True,
        load_default={},
    )
    architectureType = ma.fields.Str(
        validate=ma.validate.OneOf(
            [model_framework.value for model_framework in ModelType],
            error="`architectureType` must be one of the supported frameworks. "
            + "Check out our API reference for a full list"
            + " https://reference.openlayer.com/reference/api/openlayer.ModelType.html.\n ",
        ),
        required=True,
    )
