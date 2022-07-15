from marshmallow import (
    fields,
    Schema,
    validate,
    ValidationError,
    validates_schema,
)


class ProjectSchema(Schema):
    name = name = fields.Str(
        required=True,
        validate=validate.Length(
            min=1,
            max=64,
        ),
    )
    description = fields.Str(
        validate=validate.Length(
            min=1,
            max=140,
        ),
    )


class DatasetSchema(Schema):
    file_path = fields.Str()
    name = fields.Str(
        required=True,
        validate=validate.Length(
            min=1,
            max=64,
        ),
    )
    description = fields.Str(
        validate=validate.Length(
            min=1,
            max=140,
        ),
    )
    task_type = fields.Str(
        required=True,
        error_messages={
            "invalid": "`task_type` is not valid. Make sure you are importing TaskType correctly."
        },
        validate=validate.OneOf(
            ["text-classification", "tabular-classification"],
            error=f"The `task_type` must be one of either TaskType.TextClassification or TaskType.TabularClassification.",
        ),
    )
    tag_column_name = fields.List(
        fields.Str(),
        allow_none=True,
    )
    class_names = fields.List(
        fields.Str(),
    )
    label_column_name = fields.Str()
    language = fields.Str(
        default="en",
        validate=validate.Regexp(
            r"^[a-z]{2}(-[A-Z]{2})?$",
            error="The `language` of the dataset is not in the ISO 639-1 (alpha-2 code) format.",
        ),
    )
    sep = fields.Str()
    feature_names = fields.List(
        fields.Str(),
        allow_none=True,
    )
    text_column_name = fields.Str(
        allow_none=True,
    )
    categorical_feature_names = fields.List(
        fields.Str(),
    )

    @validates_schema
    def validates_label_column_not_in_feature_names(self, data, **kwargs):
        """Validates whether the label column name is not on the feature names list"""
        if data["label_column_name"] in data["feature_names"]:
            raise ValidationError(
                f"The `label_column_name` {data['label_column_name']} must not be in `feature_names`."
            )

    @validates_schema
    def validates_task_type_and_data_column(self, data, **kwargs):
        """Validates whether the data columns are present according to the task type"""
        if (
            data["task_type"] == "tabular-classification"
            and "feature_names" not in data
        ):
            raise ValidationError(
                "Must specify `feature_names` for TabularClassification."
            )
        elif data["task_type"] == "text-classification" and "text_column" not in data:
            raise ValidationError("Must specify `text_column` for TextClassification.")


class ModelSchema(Schema):
    name = fields.Str(
        required=True,
        validate=validate.Length(
            min=1,
            max=64,
        ),
    )
    description = fields.Str(
        required=True,
        validate=validate.Length(
            min=1,
            max=140,
        ),
    )
    task_type = fields.Str(
        required=True,
        error_messages={
            "invalid": "`task_type` is not valid. Make sure you are importing TaskType correctly."
        },
        validate=validate.OneOf(
            ["text-classification", "tabular-classification"],
            error=f"The `task_type` must be one of either TaskType.TextClassification or TaskType.TabularClassification.",
        ),
    )
    model_type = fields.Str(
        required=True,
        error_messages={
            "invalid": "`model_type` is not valid. Make sure you are importing ModelType correctly."
        },
        validate=validate.OneOf(
            ["SklearnModelArtifact", "hugging-face"],
            error=f"The `model_type` must be one of either ModelType.x or ModelType.y.",
        ),
    )
    class_names = fields.List(
        fields.Str(),
    )
    requirements_txt_file = fields.Str(
        allow_none=True,
    )
    train_sample_label_column_name = fields.Str()
    feature_names = fields.List(
        fields.Str(),
        allow_none=True,
    )
    categorical_feature_names = fields.List(
        fields.Str(),
    )
    setup_script = fields.Str(
        allow_none=True,
    )
    custom_model_code = fields.Str(
        allow_none=True,
    )
    dependent_dir = fields.Str(
        allow_none=True,
    )

    @validates_schema
    def validate_custom_model_code(self, data, **kwargs):
        """Validates the model type when `custom_code` is specified"""
        if data["model_type"] == "custom" and data["custom_model_code"] is None:
            raise ValidationError(
                "`model_type` must be ModelType.custom if specifying `custom_model_code`."
            )
        elif data["custom_model_code"] is not None and data["model_type"] != "custom":
            raise ValidationError(
                "Must specify `custom_model_code` when using ModelType.custom"
            )

    @validates_schema
    def validate_custom_model_dependent_dir(self, data, **kwargs):
        if data["model_type"] == "custom" and data["dependent_dir"] is None:
            raise ValidationError(
                "Must specify `dependent_dir` when using ModelType.custom"
            )

    @validates_schema
    def validate_custom_model_requirements(self, data, **kwargs):
        if data["model_type"] == "custom" and data["requirements_txt_file"] is None:
            raise ValidationError(
                "Must specify `requirements_txt_file` when using ModelType.custom"
            )
