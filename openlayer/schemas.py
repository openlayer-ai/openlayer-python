import marshmallow as ma

from .datasets import DatasetType
from .models import ModelType


class ProjectSchema(ma.Schema):
    name = ma.fields.Str(
        required=True,
        validate=ma.validate.Length(
            min=1,
            max=64,
        ),
    )
    description = ma.fields.Str(
        validate=ma.validate.Length(
            min=1,
            max=140,
        ),
        allow_none=True,
    )


class ModelSchema(ma.Schema):
    name = ma.fields.Str(
        required=True,
        validate=ma.validate.Length(
            min=1,
            max=64,
        ),
    )
    model_type = ma.fields.Str(
        validate=ma.validate.OneOf(
            [model_framework.value for model_framework in ModelType],
            error=f"`model_type` must be one of the supported frameworks. Check out our API reference for a full list https://reference.openlayer.com/reference/api/openlayer.ModelType.html.\n ",
        ),
        allow_none=True,
    )
    class_names = ma.fields.List(
        ma.fields.Str(),
    )
    feature_names = ma.fields.List(
        ma.fields.Str(),
        allow_none=True,
    )
    categorical_feature_names = ma.fields.List(
        ma.fields.Str(),
    )


class DatasetSchema(ma.Schema):
    file_path = ma.fields.Str()
    commit_message = ma.fields.Str(
        validate=ma.validate.Length(
            min=1,
            max=140,
        ),
    )
    dataset_type = ma.fields.Str(
        validate=ma.validate.OneOf(
            [dataset_type.value for dataset_type in DatasetType],
            error=f"`dataset_type` must be one of the supported frameworks. Check out our API reference for a full list https://reference.openlayer.com/reference/api/openlayer.DatasetType.html.\n ",
        ),
    )
    class_names = ma.fields.List(
        ma.fields.Str(),
    )
    label_column_name = ma.fields.Str()
    language = ma.fields.Str(
        default="en",
        validate=ma.validate.Regexp(
            r"^[a-z]{2}(-[A-Z]{2})?$",
            error="`language` of the dataset is not in the ISO 639-1 (alpha-2 code) format.",
        ),
    )
    sep = ma.fields.Str()
    feature_names = ma.fields.List(
        ma.fields.Str(),
    )
    text_column_name = ma.fields.Str(
        allow_none=True,
    )
    categorical_feature_names = ma.fields.List(
        ma.fields.Str(),
    )

    @ma.validates_schema
    def validates_label_column_not_in_feature_names(self, data, **kwargs):
        """Validates whether the label column name is not on the feature names list"""
        if data["label_column_name"] in data["feature_names"]:
            raise ma.ValidationError(
                f"`label_column_name` `{data['label_column_name']}` must not be in `feature_names`."
            )
