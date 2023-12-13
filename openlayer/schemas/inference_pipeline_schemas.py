# pylint: disable=invalid-name, unused-argument
"""Schemas for the inference pipeline object that shall be created on the Openlayer
platform.
"""
import marshmallow as ma


# ---------------------------- Inference pipeline ---------------------------- #
class InferencePipelineSchema(ma.Schema):
    """Schema for inference pipelines."""

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
