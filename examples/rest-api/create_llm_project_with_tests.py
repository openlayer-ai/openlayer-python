"""Example demonstrating how to create an LLM project and add tests to it.

This script shows:
- Creating an LLM project
- Adding various types of tests suitable for LLM monitoring:
  - LLM Rubric tests (evaluating quality criteria)
  - Character length tests (checking response length)
  - PII detection tests (ensuring no personal information leakage)
  - Row count tests (monitoring data volume)
- Listing tests to verify creation
- Displaying test details

Requirements:
    - Set OPENLAYER_API_KEY environment variable
"""

import os
from typing import Any

from openlayer import Openlayer
from openlayer.types.project_create_response import ProjectCreateResponse
from openlayer.types.projects.test_create_response import TestCreateResponse
from openlayer.types.projects.test_list_response import TestListResponse


def create_llm_project(client: Openlayer) -> ProjectCreateResponse:
    """Create an LLM-based project.

    Args:
        client: Initialized Openlayer client.

    Returns:
        ProjectCreateResponse containing project details.
    """
    print("\n" + "=" * 70)
    print("Step 1: Creating LLM Project")
    print("=" * 70)

    project = client.projects.create(
        name="CX Chatbot",
        task_type="llm-base",
        description="LLM project for monitoring customer support chatbot quality and safety",
    )

    print(f"✅ Created LLM project: {project.name}")
    print(f"   Project ID: {project.id}")
    print(f"   Task Type: {project.task_type}")
    print(f"   App Link: {project.links.app}")

    return project


def create_llm_rubric_test(client: Openlayer, project_id: str) -> TestCreateResponse:
    """Create an LLM rubric test to evaluate response quality.

    This test uses LLM-based evaluation to check if responses meet quality criteria.

    Args:
        client: Initialized Openlayer client.
        project_id: The ID of the project to add the test to.

    Returns:
        TestCreateResponse containing test details.
    """
    print("\n📝 Creating LLM Rubric Test...")

    test = client.projects.tests.create(
        project_id=project_id,
        name="Response Quality Rubric",
        description="Evaluate if chatbot responses are helpful, accurate, and professional",
        type="performance",
        subtype="llmRubricThresholdV2",
        thresholds=[
            {
                "insight_name": "llmRubricV2",
                "measurement": "passRate",
                "operator": ">=",
                "value": 0.8,
                "threshold_mode": "manual",
                "insight_parameters": [
                    {
                        "name": "rubric",
                        "value": "The response should be helpful, accurate, and maintain a professional tone.",
                    }
                ],
            }
        ],
        uses_ml_model=True,
        uses_production_data=False,
        uses_reference_dataset=False,
        uses_training_dataset=False,
        uses_validation_dataset=False,
    )

    print(f"✅ Created test: {test.name}")
    print(f"   Test ID: {test.id}")
    print(f"   Type: {test.type}")
    print(f"   Subtype: {test.subtype}")

    return test


def create_character_length_test(client: Openlayer, project_id: str) -> TestCreateResponse:
    """Create a character length test to ensure responses aren't too short or too long.

    Args:
        client: Initialized Openlayer client.
        project_id: The ID of the project to add the test to.

    Returns:
        TestCreateResponse containing test details.
    """
    print("\n📏 Creating Character Length Test...")

    test = client.projects.tests.create(
        project_id=project_id,
        name="Response Length Check",
        description="Ensure responses are between 50 and 2000 characters",
        type="integrity",
        subtype="characterLength",
        thresholds=[
            {
                "insight_name": "characterLength",
                "measurement": "averageCharacterLength",
                "operator": ">=",
                "value": 50.0,
                "threshold_mode": "manual",
                "insight_parameters": [{"name": "column_name", "value": "output"}],
            },
            {
                "insight_name": "characterLength",
                "measurement": "averageCharacterLength",
                "operator": "<=",
                "value": 2000.0,
                "threshold_mode": "manual",
                "insight_parameters": [{"name": "column_name", "value": "output"}],
            },
        ],
        uses_ml_model=False,
        uses_production_data=False,
        uses_reference_dataset=False,
        uses_training_dataset=False,
        uses_validation_dataset=False,
    )

    print(f"✅ Created test: {test.name}")
    print(f"   Test ID: {test.id}")
    print(f"   Type: {test.type}")
    print(f"   Subtype: {test.subtype}")

    return test


def create_pii_detection_test(client: Openlayer, project_id: str) -> TestCreateResponse:
    """Create a PII detection test to ensure no personal information is leaked.

    Args:
        client: Initialized Openlayer client.
        project_id: The ID of the project to add the test to.

    Returns:
        TestCreateResponse containing test details.
    """
    print("\n🔒 Creating PII Detection Test...")

    test = client.projects.tests.create(
        project_id=project_id,
        name="PII Leakage Prevention",
        description="Ensure chatbot responses don't contain personal identifiable information",
        type="integrity",
        subtype="containsPii",
        thresholds=[
            {
                "insight_name": "containsPii",
                "measurement": "piiRowCount",
                "operator": "is",
                "value": 0.0,
                "threshold_mode": "manual",
                "insight_parameters": [{"name": "column_name", "value": "output"}],
            }
        ],
        uses_ml_model=False,
        uses_production_data=False,
        uses_reference_dataset=False,
        uses_training_dataset=False,
        uses_validation_dataset=False,
    )

    print(f"✅ Created test: {test.name}")
    print(f"   Test ID: {test.id}")
    print(f"   Type: {test.type}")
    print(f"   Subtype: {test.subtype}")

    return test


def create_row_count_test(
    client: Openlayer, project_id: str, monitoring: bool = True
) -> TestCreateResponse:
    """Create a row count test to monitor data volume.

    Args:
        client: Initialized Openlayer client.
        project_id: The ID of the project to add the test to.
        monitoring: Whether this is a monitoring test (uses production data).

    Returns:
        TestCreateResponse containing test details.
    """
    print("\n📊 Creating Row Count Test...")

    test_config: dict[str, Any] = {
        "project_id": project_id,
        "name": "Minimum Daily Interactions",
        "description": "Ensure we have sufficient data for monitoring (at least 100 rows per day)",
        "type": "integrity",
        "subtype": "rowCount",
        "thresholds": [
            {
                "insight_name": "metrics",
                "measurement": "rowCount",
                "operator": ">=",
                "value": 100.0,
                "threshold_mode": "manual",
            }
        ],
    }

    if monitoring:
        # For monitoring mode, add production data parameters
        test_config["uses_production_data"] = True
        test_config["uses_reference_dataset"] = False
        test_config["evaluation_window"] = 86400.0  # 24 hours in seconds
        test_config["delay_window"] = 3600.0  # 1 hour in seconds
    else:
        test_config["uses_production_data"] = False
        test_config["uses_reference_dataset"] = False

    # Add common required fields
    test_config["uses_ml_model"] = False
    test_config["uses_training_dataset"] = False
    test_config["uses_validation_dataset"] = False

    test = client.projects.tests.create(**test_config)

    print(f"✅ Created test: {test.name}")
    print(f"   Test ID: {test.id}")
    print(f"   Type: {test.type}")
    print(f"   Subtype: {test.subtype}")
    if monitoring:
        print(f"   Uses Production Data: Yes")
        print(f"   Evaluation Window: {test.evaluation_window}s (24 hours)")

    return test


def create_sentence_length_test(client: Openlayer, project_id: str) -> TestCreateResponse:
    """Create a sentence length test to ensure responses are concise.

    Args:
        client: Initialized Openlayer client.
        project_id: The ID of the project to add the test to.

    Returns:
        TestCreateResponse containing test details.
    """
    print("\n📝 Creating Sentence Length Test...")

    test = client.projects.tests.create(
        project_id=project_id,
        name="Sentence Length Check",
        description="Ensure responses have appropriate sentence length (not too verbose)",
        type="integrity",
        subtype="sentenceLength",
        thresholds=[
            {
                "insight_name": "sentenceLength",
                "measurement": "averageWordsPerSentence",
                "operator": "<=",
                "value": 30.0,
                "threshold_mode": "manual",
                "insight_parameters": [{"name": "column_name", "value": "output"}],
            }
        ],
        uses_ml_model=False,
        uses_production_data=False,
        uses_reference_dataset=False,
        uses_training_dataset=False,
        uses_validation_dataset=False,
    )

    print(f"✅ Created test: {test.name}")
    print(f"   Test ID: {test.id}")
    print(f"   Type: {test.type}")
    print(f"   Subtype: {test.subtype}")

    return test


def list_project_tests(client: Openlayer, project_id: str) -> TestListResponse:
    """List all tests for a project.

    Args:
        client: Initialized Openlayer client.
        project_id: The ID of the project to list tests for.

    Returns:
        TestListResponse containing list of tests.
    """
    print("\n" + "=" * 70)
    print("Step 2: Listing All Tests")
    print("=" * 70)

    tests = client.projects.tests.list(project_id=project_id)

    print(f"\n📋 Found {len(tests.items)} test(s) in the project:")
    for test in tests.items:
        print(f"\n   • {test.name}")
        print(f"     ID: {test.id}")
        print(f"     Type: {test.type}")
        print(f"     Subtype: {test.subtype}")
        print(f"     Description: {test.description}")

    return tests


def display_test_summary(tests: TestListResponse) -> None:
    """Display a summary of tests by type.

    Args:
        tests: TestListResponse containing list of tests.
    """
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)

    # Count tests by type
    test_types: dict[str, int] = {}
    test_subtypes: dict[str, int] = {}

    for test in tests.items:
        test_types[test.type] = test_types.get(test.type, 0) + 1
        test_subtypes[test.subtype] = test_subtypes.get(test.subtype, 0) + 1

    print("\n📊 Tests by Type:")
    for test_type, count in test_types.items():
        print(f"   {test_type}: {count}")

    print("\n🔍 Tests by Subtype:")
    for subtype, count in test_subtypes.items():
        print(f"   {subtype}: {count}")


def main() -> None:
    """Main function demonstrating LLM project creation with tests.

    This function demonstrates:
    - Creating an LLM project
    - Adding various types of tests suitable for LLM monitoring
    - Listing and summarizing created tests
    """
    # Initialize the Openlayer client
    client = Openlayer(
        api_key=os.environ.get("OPENLAYER_API_KEY")
    )

    print("=" * 70)
    print("Openlayer: Create LLM Project with Tests")
    print("=" * 70)

    try:
        # Step 1: Create LLM project
        project = create_llm_project(client)

        # Step 2: Create various tests for the project
        print("\n" + "=" * 70)
        print("Step 2: Creating Tests")
        print("=" * 70)

        # Create LLM-specific rubric test
        create_llm_rubric_test(client, project.id)

        # Create character length test
        create_character_length_test(client, project.id)

        # Create PII detection test
        create_pii_detection_test(client, project.id)

        # Create sentence length test
        create_sentence_length_test(client, project.id)

        # Create row count test (monitoring mode)
        create_row_count_test(client, project.id, monitoring=True)

        # Step 3: List all tests
        tests = list_project_tests(client, project.id)

        # Step 4: Display summary
        display_test_summary(tests)

        print("\n" + "=" * 70)
        print("✅ Successfully created LLM project with tests!")
        print("=" * 70)
        print(f"\n🔗 View your project at: {project.links.app}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()

