# pylint: disable=invalid-name, unused-argument
"""Schemas for the data configs that shall be uploaded to the Openlayer platform.
"""
import marshmallow as ma
import marshmallow_oneofschema as maos

from .. import constants
from ..datasets import DatasetType
from ..tasks import TaskType


# ----------- Development datasets (i.e., training and validation) ----------- #
class BaseDevelopmentDatasetSchema(ma.Schema):
    """Common schema for development datasets for all task types."""

    columnNames = ma.fields.List(
        ma.fields.Str(validate=constants.COLUMN_NAME_VALIDATION_LIST),
        allow_none=True,
        load_default=None,
    )
    label = ma.fields.Str(
        validate=ma.validate.OneOf(
            [DatasetType.Training.value, DatasetType.Validation.value],
            error="`label` not supported."
            + "The supported `labels` are 'training', 'validation'.",
        ),
        required=True,
    )
    language = ma.fields.Str(
        load_default="en",
        validate=constants.LANGUAGE_CODE_REGEX,
    )
    metadata = ma.fields.Dict(allow_none=True, load_default={})
    sep = ma.fields.Str(load_default=",")


class LLMInputSchema(ma.Schema):
    """Specific schema for the input part of LLM datasets."""

    inputVariableNames = ma.fields.List(
        ma.fields.Str(validate=constants.COLUMN_NAME_VALIDATION_LIST), required=True
    )
    contextColumnName = ma.fields.Str(
        validate=constants.COLUMN_NAME_VALIDATION_LIST,
        allow_none=True,
        load_default=None,
    )
    questionColumnName = ma.fields.Str(
        validate=constants.COLUMN_NAME_VALIDATION_LIST,
        allow_none=True,
        load_default=None,
    )


class TabularInputSchema(ma.Schema):
    """Specific schema for tabular datasets."""

    categoricalFeatureNames = ma.fields.List(
        ma.fields.Str(validate=constants.COLUMN_NAME_VALIDATION_LIST),
        allow_none=True,
        load_default=[],
    )
    featureNames = ma.fields.List(
        ma.fields.Str(validate=constants.COLUMN_NAME_VALIDATION_LIST),
        load_default=[],
    )


class TextInputSchema(ma.Schema):
    """Specific schema for text datasets."""

    textColumnName = ma.fields.Str(
        validate=constants.COLUMN_NAME_VALIDATION_LIST,
        allow_none=True,
    )


class ClassificationOutputSchema(ma.Schema):
    """Specific schema for classification datasets."""

    classNames = ma.fields.List(ma.fields.Str(), required=True)
    labelColumnName = ma.fields.Str(
        validate=constants.COLUMN_NAME_VALIDATION_LIST,
        allow_none=True,
        load_default=None,
    )
    predictionsColumnName = ma.fields.Str(
        validate=constants.COLUMN_NAME_VALIDATION_LIST,
        allow_none=True,
        load_default=None,
    )
    predictionScoresColumnName = ma.fields.Str(
        validate=constants.COLUMN_NAME_VALIDATION_LIST,
        allow_none=True,
        load_default=None,
    )


class LLMOutputSchema(ma.Schema):
    """Specific schema for the output part of LLM datasets."""

    groundTruthColumnName = ma.fields.Str(
        validate=constants.COLUMN_NAME_VALIDATION_LIST,
        allow_none=True,
        load_default=None,
    )
    costColumnName = ma.fields.Str(
        validate=constants.COLUMN_NAME_VALIDATION_LIST,
        allow_none=True,
        load_default=None,
    )
    numOfTokenColumnName = ma.fields.Str(
        validate=constants.COLUMN_NAME_VALIDATION_LIST,
        allow_none=True,
        load_default=None,
    )
    outputColumnName = ma.fields.Str(
        validate=constants.COLUMN_NAME_VALIDATION_LIST,
        allow_none=True,
        load_default=None,
    )


class RegressionOutputSchema(ma.Schema):
    """Specific schema for regression datasets."""

    targetColumnName = ma.fields.Str(
        validate=constants.COLUMN_NAME_VALIDATION_LIST,
        allow_none=True,
        load_default=None,
    )
    predictionsColumnName = ma.fields.Str(
        validate=constants.COLUMN_NAME_VALIDATION_LIST,
        allow_none=True,
        load_default=None,
    )


class LLMDatasetSchema(BaseDevelopmentDatasetSchema, LLMInputSchema, LLMOutputSchema):
    """LLM dataset schema."""

    # Overwrite the label to allow for a 'fine-tuning' label instead
    # of the 'training' label
    label = ma.fields.Str(
        validate=ma.validate.OneOf(
            [
                DatasetType.FineTuning.value,
                DatasetType.Validation.value,
            ],
            error="`label` not supported."
            + "The supported `labels` are 'fine-tuning' and 'validation'.",
        ),
        required=True,
    )


class TabularClassificationDatasetSchema(
    BaseDevelopmentDatasetSchema, TabularInputSchema, ClassificationOutputSchema
):
    """Tabular classification dataset schema."""

    pass


class TabularRegressionDatasetSchema(
    BaseDevelopmentDatasetSchema, TabularInputSchema, RegressionOutputSchema
):
    """Tabular regression dataset schema."""

    pass


class TextClassificationDatasetSchema(
    BaseDevelopmentDatasetSchema, TextInputSchema, ClassificationOutputSchema
):
    """Text classification dataset schema."""

    pass


class DatasetSchema(maos.OneOfSchema):
    """One of schema for dataset. Returns the correct schema based on the task type."""

    type_field = "task_type"
    type_schemas = {
        TaskType.TabularClassification.value: TabularClassificationDatasetSchema,
        TaskType.TabularRegression.value: TabularRegressionDatasetSchema,
        TaskType.TextClassification.value: TextClassificationDatasetSchema,
        TaskType.LLM.value: LLMDatasetSchema,
        TaskType.LLMNER.value: LLMDatasetSchema,
        TaskType.LLMQuestionAnswering.value: LLMDatasetSchema,
        TaskType.LLMSummarization.value: LLMDatasetSchema,
        TaskType.LLMTranslation.value: LLMDatasetSchema,
    }

    def get_obj_type(self, obj):
        if obj not in [task_type.value for task_type in TaskType]:
            raise ma.ValidationError(f"Unknown object type: {obj.__class__.__name__}")
        return obj


# ---------------------------- Reference datasets ---------------------------- #
class LLMReferenceDatasetSchema(LLMDatasetSchema):
    """LLM reference dataset schema."""

    # Overwrite the label to allow for a 'reference' label instead
    label = ma.fields.Str(
        validate=ma.validate.OneOf(
            [DatasetType.Reference.value],
            error="`label` not supported." + "The supported `labels` are 'reference'.",
        ),
        required=True,
    )


class TabularClassificationReferenceDatasetSchema(TabularClassificationDatasetSchema):
    """Tabular classification reference dataset schema."""

    # Overwrite the label to allow for a 'reference' label instead
    label = ma.fields.Str(
        validate=ma.validate.OneOf(
            [DatasetType.Reference.value],
            error="`label` not supported." + "The supported `labels` are 'reference'.",
        ),
        required=True,
    )


class TabularRegressionReferenceDatasetSchema(TabularRegressionDatasetSchema):
    """Tabular regression reference dataset schema."""

    # Overwrite the label to allow for a 'reference' label instead
    label = ma.fields.Str(
        validate=ma.validate.OneOf(
            [DatasetType.Reference.value],
            error="`label` not supported." + "The supported `labels` are 'reference'.",
        ),
        required=True,
    )


class TextClassificationReferenceDatasetSchema(TextClassificationDatasetSchema):
    """Text classification reference dataset schema."""

    # Overwrite the label to allow for a 'reference' label instead
    label = ma.fields.Str(
        validate=ma.validate.OneOf(
            [DatasetType.Reference.value],
            error="`label` not supported." + "The supported `labels` are 'reference'.",
        ),
        required=True,
    )


class ReferenceDatasetSchema(maos.OneOfSchema):
    """One of schema for reference datasets.
    Returns the correct schema based on the task type."""

    type_field = "task_type"
    # pylint: disable=line-too-long
    type_schemas = {
        TaskType.TabularClassification.value: TabularClassificationReferenceDatasetSchema,
        TaskType.TabularRegression.value: TabularRegressionReferenceDatasetSchema,
        TaskType.TextClassification.value: TextClassificationReferenceDatasetSchema,
        TaskType.LLM.value: LLMReferenceDatasetSchema,
        TaskType.LLMNER.value: LLMReferenceDatasetSchema,
        TaskType.LLMQuestionAnswering.value: LLMReferenceDatasetSchema,
        TaskType.LLMSummarization.value: LLMReferenceDatasetSchema,
        TaskType.LLMTranslation.value: LLMReferenceDatasetSchema,
    }

    def get_obj_type(self, obj):
        if obj not in [task_type.value for task_type in TaskType]:
            raise ma.ValidationError(f"Unknown object type: {obj.__class__.__name__}")
        return obj


# ------------------------------ Production data ----------------------------- #
class BaseProductionDataSchema(ma.Schema):
    """Common schema for production datasets for all task types."""

    inferenceIdColumnName = ma.fields.Str(
        validate=constants.COLUMN_NAME_VALIDATION_LIST,
        allow_none=True,
        load_default=None,
    )
    latencyColumnName = ma.fields.Str(
        validate=constants.COLUMN_NAME_VALIDATION_LIST,
        allow_none=True,
        load_default=None,
    )
    metadata = ma.fields.Dict(allow_none=True, load_default={})
    timestampColumnName = ma.fields.Str(
        validate=constants.COLUMN_NAME_VALIDATION_LIST,
        allow_none=True,
        load_default=None,
    )
    label = ma.fields.Str(
        validate=ma.validate.OneOf(
            [DatasetType.Production.value],
            error="`label` not supported." + "The supported label is 'production'.",
        ),
        required=True,
    )


class LLMProductionDataSchema(
    BaseProductionDataSchema, LLMInputSchema, LLMOutputSchema
):
    """LLM production data schema."""

    prompt = ma.fields.List(ma.fields.Dict(), load_default=None)

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


class TabularClassificationProductionDataSchema(
    BaseProductionDataSchema, TabularInputSchema, ClassificationOutputSchema
):
    """Tabular classification production data schema."""

    pass


class TabularRegressionProductionDataSchema(
    BaseProductionDataSchema, TabularInputSchema, RegressionOutputSchema
):
    """Tabular regression production data schema."""

    pass


class TextClassificationProductionDataSchema(
    BaseProductionDataSchema, TextInputSchema, ClassificationOutputSchema
):
    """Text classification production data schema."""

    pass


class ProductionDataSchema(maos.OneOfSchema):
    """One of schema for production data. Returns the correct schema based on
    the task type."""

    type_field = "task_type"
    type_schemas = {
        TaskType.TabularClassification.value: TabularClassificationProductionDataSchema,
        TaskType.TabularRegression.value: TabularRegressionProductionDataSchema,
        TaskType.TextClassification.value: TextClassificationProductionDataSchema,
        TaskType.LLM.value: LLMProductionDataSchema,
        TaskType.LLMNER.value: LLMProductionDataSchema,
        TaskType.LLMQuestionAnswering.value: LLMProductionDataSchema,
        TaskType.LLMSummarization.value: LLMProductionDataSchema,
        TaskType.LLMTranslation.value: LLMProductionDataSchema,
    }

    def get_obj_type(self, obj):
        if obj not in [task_type.value for task_type in TaskType]:
            raise ma.ValidationError(f"Unknown object type: {obj.__class__.__name__}")
        return obj
