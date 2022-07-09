from marshmallow import (
    fields,
    Schema,
    validate,
    ValidationError,
    validates_schema,
)


class DatasetSchema(Schema):
    file_path = fields.Str(
        error_messages={"invalid": "The `file_path` is not a valid string."}
    )
    name = fields.Str(
        required=True,
        error_messages={"invalid": "The `name` is not a valid string."},
        validate=validate.Length(
            min=1,
            max=64,
            error=f"`name` string must be at least 1 and at most 64 characters long.",
        ),
    )
    description = fields.Str(
        required=True,
        error_messages={"invalid": "`description` is not a valid string."},
        validate=validate.Length(
            min=1,
            max=140,
            error=f"The `description` string must be at least 1 and at most 140 characters long.",
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
        error_messages={"invalid": "`tag_column_name` is not a valid list of strings."},
        allow_none=True,
    )
    class_names = fields.List(
        fields.Str(),
        error_messages={"invalid": "`class_names` is not a valid list of strings."},
    )
    label_column_name = fields.Str(
        error_messages={"invalid": "The `label_column_name` is not a valid string."}
    )
    language = fields.Str(
        default="en",
        error_messages={"invalid": "`language` is not a valid string."},
        validate=validate.Regexp(
            r"^[a-z]{2}(-[A-Z]{2})?$",
            error="The `language` of the dataset is not in the ISO 639-1 (alpha-2 code) format.",
        ),
    )
    sep = fields.Str(error_messages={"invalid": "The `sep` is not a valid string."})
    feature_names = fields.List(
        fields.Str(),
        error_messages={"invalid": "`feature_names` is not a valid list of strings."},
        allow_none=True,
    )
    text_column_name = fields.Str(
        error_messages={"invalid": "`text_column_name` is not a valid string."},
        allow_none=True,
    )
    categorical_feature_names = fields.List(
        fields.Str(),
        error_messages={
            "invalid": "The `categorical_feature_names` is not a valid list of strings."
        },
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
                "Must specify `feature_names` for TabularClassification"
            )
        elif data["task_type"] == "text-classification" and "text_column" not in data:
            raise ValidationError("Must specify `text_column` for TextClassification")


class ModelSchema(Schema):
    name = fields.Str(
        required=True,
        error_messages={"invalid": "The `name` is not a valid string."},
        validate=validate.Length(
            min=1,
            max=64,
            error=f"`name` string must be at least 1 and at most 64 characters long.",
        ),
    )
    function = fields.Function(
        required=True,
        error_messages={"invalid": "The `function` is not a valid function"},
    )
    description = fields.Str(
        required=True,
        error_messages={"invalid": "`description` is not a valid string."},
        validate=validate.Length(
            min=1,
            max=140,
            error=f"The `description` string must be at least 1 and at most 140 characters long.",
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
            ["scikit-learn", "hugging-face"],
            error=f"The `model_type` must be one of either ModelType.x or ModelType.y.",
        ),
    )
    class_names = fields.List(
        fields.Str(),
        error_messages={"invalid": "`class_names` is not a valid list of strings."},
    )
    requirements_txt_file = fields.Str(
        error_messages={
            "invalid": "The `requirements_txt_file` is not a valid string."
        },
        allow_none=True,
    )
    train_sample_label_column_name = fields.Str(
        error_messages={
            "invalid": "The `train_sample_label_column_name` is not a valid string."
        }
    )
    feature_names = fields.List(
        fields.Str(),
        error_messages={"invalid": "`feature_names` is not a valid list of strings."},
        allow_none=True,
    )
    categorical_feature_names = fields.List(
        fields.Str(),
        error_messages={
            "invalid": "The `categorical_feature_names` is not a valid list of strings."
        },
    )
    setup_script = fields.Str(
        error_messages={"invalid": "The `setup_script` is not a valid string."},
        allow_none=True,
    )
    custom_model_code = fields.Str(
        error_messages={"invalid": "The `custom_model_code` is not a valid string."},
        allow_none=True,
    )
    dependent_dir = fields.Str(
        error_messages={"invalid": "The `dependent_dir` is not a valid string."},
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

    @validates_schema
    def validate_task_type_args(self, data, **kwargs):
        if data["task_type"] in ["tabular-classification", "tabular-regression"]:
            required_fields = [
                (data["feature_names"], "feature_names"),
                (data["train_sample_df"], "train_sample_df"),
                (
                    data["train_sample_label_column_name"],
                    "train_sample_label_column_name",
                ),
            ]
            for value, field in required_fields:
                if value is None:
                    raise ValidationError(
                        f"Must specify {field} for TabularClassification"
                    )
