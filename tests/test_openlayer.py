"""
Module with sample openlayer test
"""
import openlayer


def test_openlayer():
    assert openlayer.api.OPENLAYER_ENDPOINT == "https://api.openlayer.com/v1"
