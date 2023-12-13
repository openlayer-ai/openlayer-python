# pylint: disable=invalid-name, unused-argument
"""Schemas for the model configs that shall be uploaded to the Openlayer platform.
"""
import marshmallow as ma
import marshmallow_oneofschema as maos

from .. import constants
from ..models import ModelType
from ..tasks import TaskType


# ---------------------------------- Models ---------------------------------- #
class BaseModelSchema(ma.Schema):
    """Common schema for models for all task types."""

    name = ma.fields.Str(
        validate=ma.validate.Length(
            min=1,
            max=64,
        ),
        allow_none=True,
        load_default="Model",
    )
    metadata = ma.fields.Dict(
        allow_none=True,
        load_default={},
    )
    modelType = ma.fields.Str()
    architectureType = ma.fields.Str(
        validate=ma.validate.OneOf(
            [model_framework.value for model_framework in ModelType],
            error="`architectureType` must be one of the supported frameworks."
            + " Check out our API reference for a full list."
            + " If you can't find your framework, specify 'custom' for your model's"
            + " `architectureType`.",
        ),
        allow_none=True,
        load_default="custom",
    )


class TabularModelSchema(ma.Schema):
    """Specific schema for tabular models."""

    categoricalFeatureNames = ma.fields.List(
        ma.fields.Str(validate=constants.COLUMN_NAME_VALIDATION_LIST),
        allow_none=True,
        load_default=[],
    )
    featureNames = ma.fields.List(
        ma.fields.Str(validate=constants.COLUMN_NAME_VALIDATION_LIST),
        load_default=[],
    )


class ClassificationModelSchema(ma.Schema):
    """Specific schema for classification models."""

    classNames = ma.fields.List(
        ma.fields.Str(),
        required=True,
    )
    predictionThreshold = ma.fields.Float(
        allow_none=True,
        validate=ma.validate.Range(
            min=0.0,
            max=1.0,
        ),
        load_default=None,
    )

    @ma.validates_schema
    def validates_prediction_threshold_and_class_names(self, data, **kwargs):
        """Validates whether a prediction threshold was specified for a
        binary classification model."""
        if data["predictionThreshold"] and len(data["classNames"]) != 2:
            raise ma.ValidationError(
                "`predictionThreshold` can only be specified for binary classification models."
            )


class LLMModelSchema(BaseModelSchema):
    """Specific schema for LLM models."""

    prompt = ma.fields.List(ma.fields.Dict())
    model = ma.fields.Str()
    modelProvider = ma.fields.Str()
    modelParameters = ma.fields.Dict()
    inputVariableNames = ma.fields.List(
        ma.fields.Str(validate=constants.COLUMN_NAME_VALIDATION_LIST),
        load_default=[],
    )
    # Important that here the architectureType defaults to `llm` and not `custom` since
    # the architectureType is used to check if the model is an LLM or not.
    architectureType = ma.fields.Str(
        validate=ma.validate.OneOf(
            [model_framework.value for model_framework in ModelType],
            error="`architectureType` must be one of the supported frameworks."
            + " Check out our API reference for a full list."
            + " If you can't find your framework, specify 'custom' for your model's"
            + " `architectureType`.",
        ),
        allow_none=True,
        load_default="llm",
    )

    @ma.validates_schema
    def validates_model_type_fields(self, data, **kwargs):
        """Validates the required fields depending on the modelType."""
        if data["modelType"] == "api":
            if (
                data.get("prompt") is None
                or data.get("modelProvider") is None
                or data.get("model") is None
            ):
                # TODO: rename "direct to API"
                raise ma.ValidationError(
                    "To use the direct to API approach for LLMs, you must "
                    "provide at least the `prompt` and specify the "
                    "`modelProvider`, and `model`."
                )

    @ma.validates_schema
    def validates_prompt(self, data, **kwargs):
        """Validates the prompt structure."""
        if data.get("prompt") is not None:
            for message in data.get("prompt"):
                if message.get("role") is None:
                    raise ma.ValidationError(
                        "Each message in the prompt must have a `role`."
                    )
                else:
                    if message.get("role") not in ["system", "user", "assistant"]:
                        raise ma.ValidationError(
                            "The `role` of each message in the prompt must be one of "
                            "'system', 'user', or 'assistant'."
                        )
                if message.get("content") is None:
                    raise ma.ValidationError(
                        "Each message in the prompt must have a `content`."
                    )
                else:
                    if not isinstance(message.get("content"), str):
                        raise ma.ValidationError(
                            "The `content` of each message in the prompt must be a string."
                        )


class TabularClassificationModelSchema(
    BaseModelSchema, TabularModelSchema, ClassificationModelSchema
):
    """Tabular classification model schema."""

    pass


class TabularRegressionModelSchema(BaseModelSchema, TabularModelSchema):
    """Tabular regression model schema."""

    pass


class TextClassificationModelSchema(BaseModelSchema, ClassificationModelSchema):
    """Text classification model schema."""

    pass


class ModelSchema(maos.OneOfSchema):
    """One of schema for models. Returns the correct schema based on the task type."""

    type_field = "task_type"
    type_schemas = {
        TaskType.TabularClassification.value: TabularClassificationModelSchema,
        TaskType.TabularRegression.value: TabularRegressionModelSchema,
        TaskType.TextClassification.value: TextClassificationModelSchema,
        TaskType.LLM.value: LLMModelSchema,
        TaskType.LLMNER.value: LLMModelSchema,
        TaskType.LLMQuestionAnswering.value: LLMModelSchema,
        TaskType.LLMSummarization.value: LLMModelSchema,
        TaskType.LLMTranslation.value: LLMModelSchema,
    }

    def get_obj_type(self, obj):
        if obj not in [task_type.value for task_type in TaskType]:
            raise ma.ValidationError(f"Unknown object type: {obj.__class__.__name__}")
        return obj


# ------------------------------ Baseline models ----------------------------- #
class BaseBaselineModelSchema(ma.Schema):
    """Common schema for baseline models for all task types."""

    metadata = ma.fields.Dict(allow_none=True, load_default={})
    modelType = ma.fields.Str()


class TabularClassificationBaselineModelSchema(BaseBaselineModelSchema):
    """Tabular classification baseline model schema."""

    pass


class BaselineModelSchema(maos.OneOfSchema):
    """Schema for baseline models."""

    type_field = "task_type"
    type_schemas = {
        "tabular-classification": TabularClassificationBaselineModelSchema,
    }

    def get_obj_type(self, obj):
        if obj != "tabular-classification":
            raise ma.ValidationError(f"Unknown object type: {obj.__class__.__name__}")
        return obj
