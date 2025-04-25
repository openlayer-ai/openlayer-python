"""Module containing convenience functions for the tests API."""

from typing import Optional, List
from openlayer import Openlayer


def copy_tests(
    client: Openlayer,
    origin_project_id: str,
    target_project_id: str,
    verbose: bool = False,
    test_ids: Optional[List[str]] = None,
) -> None:
    """Copy tests from one project to another.

    Args:
        client (Openlayer): The Openlayer client.
        origin_project_id (str): The ID of the origin project (where the tests
          are).
        target_project_id (str): The ID of the target project (where the tests
          will be copied to).
        verbose (bool): Whether to print verbose output.
        test_ids (List[str]): The IDs of the tests to copy. If not provided, all
          tests will be copied.
    """
    tests = client.projects.tests.list(project_id=origin_project_id)

    if test_ids is None and verbose:
        print("Copying all tests from the origin project to the target project.")
    else:
        print(
            "Copying the following tests from the origin project to"
            f" the target project: {test_ids}"
        )

    for test in tests.items:
        if test.id in test_ids:
            thresholds = _parse_thresholds(test.thresholds)
            client.projects.tests.create(
                project_id=target_project_id,
                name=test.name,
                description=test.description,
                type=test.type,
                subtype=test.subtype,
                thresholds=thresholds,
                uses_production_data=test.uses_production_data,
                evaluation_window=test.evaluation_window,
                delay_window=test.delay_window,
                uses_training_dataset=test.uses_training_dataset,
                uses_validation_dataset=test.uses_validation_dataset,
                uses_ml_model=test.uses_ml_model,
            )
            if verbose:
                print(
                    f"Copied test '{test.id}' - '{test.name}' from the"
                    " origin project to the target project."
                )


def _parse_thresholds(thresholds: List[dict]) -> List[dict]:
    """Parse the thresholds from the test to the format required by the create
    test endpoint."""
    thresholds = []
    for threshold in thresholds:
        current_threshold = {
            "insightName": threshold.insight_name,
            "measurement": threshold.measurement,
            "operator": threshold.operator,
            "value": threshold.value,
        }

        if threshold.get("insightParameters"):
            current_threshold["insightParameters"] = threshold["insightParameters"]
        thresholds.append(current_threshold)

    return thresholds
