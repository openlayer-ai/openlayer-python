"""Example demonstrating how to create and list projects using the Openlayer SDK.

This script shows:
- Creating projects with different task types
- Adding optional descriptions
- Listing all projects
- Filtering projects by name and task type
- Handling API responses

Requirements:
    - Set OPENLAYER_API_KEY environment variable
"""

import os
from typing import Optional

from openlayer import Openlayer
from openlayer.types.project_create_response import ProjectCreateResponse
from openlayer.types.project_list_response import ProjectListResponse


def create_llm_project(client: Openlayer, name: str, description: Optional[str] = None) -> ProjectCreateResponse:
    """Create an LLM-based project.

    Args:
        client: Initialized Openlayer client.
        name: Name for the project.
        description: Optional description for the project.

    Returns:
        ProjectCreateResponse containing project details.
    """
    print(f"\n📝 Creating LLM project: {name}")
    project = client.projects.create(
        name=name,
        task_type="llm-base",
        description=description,
    )
    print(f"✅ Created project with ID: {project.id}")
    print(f"   App link: {project.links.app}")
    return project


def create_tabular_classification_project(
    client: Openlayer, name: str, description: Optional[str] = None
) -> ProjectCreateResponse:
    """Create a tabular classification project.

    Args:
        client: Initialized Openlayer client.
        name: Name for the project.
        description: Optional description for the project.

    Returns:
        ProjectCreateResponse containing project details.
    """
    print(f"\n📊 Creating tabular classification project: {name}")
    project = client.projects.create(
        name=name,
        task_type="tabular-classification",
        description=description,
    )
    print(f"✅ Created project with ID: {project.id}")
    print(f"   App link: {project.links.app}")
    return project


def create_tabular_regression_project(
    client: Openlayer, name: str, description: Optional[str] = None
) -> ProjectCreateResponse:
    """Create a tabular regression project.

    Args:
        client: Initialized Openlayer client.
        name: Name for the project.
        description: Optional description for the project.

    Returns:
        ProjectCreateResponse containing project details.
    """
    print(f"\n📈 Creating tabular regression project: {name}")
    project = client.projects.create(
        name=name,
        task_type="tabular-regression",
        description=description,
    )
    print(f"✅ Created project with ID: {project.id}")
    print(f"   App link: {project.links.app}")
    return project


def create_text_classification_project(
    client: Openlayer, name: str, description: Optional[str] = None
) -> ProjectCreateResponse:
    """Create a text classification project.

    Args:
        client: Initialized Openlayer client.
        name: Name for the project.
        description: Optional description for the project.

    Returns:
        ProjectCreateResponse containing project details.
    """
    print(f"\n📄 Creating text classification project: {name}")
    project = client.projects.create(
        name=name,
        task_type="text-classification",
        description=description,
    )
    print(f"✅ Created project with ID: {project.id}")
    print(f"   App link: {project.links.app}")
    return project


def list_all_projects(client: Openlayer) -> ProjectListResponse:
    """List all projects in the workspace.

    Args:
        client: Initialized Openlayer client.

    Returns:
        ProjectListResponse containing list of projects.
    """
    print("\n📋 Listing all projects...")
    projects = client.projects.list()
    print(f"   Found {len(projects.items)} project(s)")
    for project in projects.items:
        print(f"   - {project.name} ({project.task_type}) - ID: {project.id}")
    return projects


def list_projects_by_name(client: Openlayer, name: str) -> ProjectListResponse:
    """List projects filtered by name.

    Args:
        client: Initialized Openlayer client.
        name: Name filter for projects.

    Returns:
        ProjectListResponse containing filtered projects.
    """
    print(f"\n🔍 Listing projects with name containing: {name}")
    projects = client.projects.list(name=name)
    print(f"   Found {len(projects.items)} project(s)")
    for project in projects.items:
        print(f"   - {project.name} ({project.task_type}) - ID: {project.id}")
    return projects


def list_projects_by_task_type(client: Openlayer, task_type: str) -> ProjectListResponse:
    """List projects filtered by task type.

    Args:
        client: Initialized Openlayer client.
        task_type: Task type filter (e.g., 'llm-base', 'tabular-classification').

    Returns:
        ProjectListResponse containing filtered projects.
    """
    print(f"\n🎯 Listing projects with task type: {task_type}")
    projects = client.projects.list(task_type=task_type)  # type: ignore
    print(f"   Found {len(projects.items)} project(s)")
    for project in projects.items:
        print(f"   - {project.name} ({project.task_type}) - ID: {project.id}")
    return projects


def display_project_details(project: ProjectCreateResponse) -> None:
    """Display detailed information about a project.

    Args:
        project: Project response object.
    """
    print(f"\n📋 Project Details:")
    print(f"   ID: {project.id}")
    print(f"   Name: {project.name}")
    print(f"   Task Type: {project.task_type}")
    print(f"   Description: {project.description or 'N/A'}")
    print(f"   Created: {project.date_created}")
    print(f"   Updated: {project.date_updated}")
    print(f"   Workspace ID: {project.workspace_id}")
    print(f"   Inference Pipelines: {project.inference_pipeline_count}")
    print(f"   Total Tests: {project.goal_count}")
    print(f"   Development Tests: {project.development_goal_count}")
    print(f"   Monitoring Tests: {project.monitoring_goal_count}")
    print(f"   Versions: {project.version_count}")
    print(f"   App Link: {project.links.app}")


def main() -> None:
    """Main function demonstrating project creation and management.

    This function demonstrates:
    - Creating projects with all supported task types
    - Using optional descriptions
    - Listing projects with various filters
    - Displaying project details
    """
    # Initialize the Openlayer client
    client = Openlayer(
        api_key=os.environ.get("OPENLAYER_API_KEY"),
    )

    print("=" * 70)
    print("Openlayer Project Creation Examples")
    print("=" * 70)

    try:
        # Example 1: Create an LLM project with description
        llm_project = create_llm_project(
            client,
            name="Customer Support Chatbot",
            description="LLM project for monitoring customer support chatbot interactions",
        )
        display_project_details(llm_project)

        # Example 2: Create a tabular classification project
        classification_project = create_tabular_classification_project(
            client,
            name="Churn Prediction Model",
            description="Predict customer churn based on usage patterns",
        )

        # Example 3: Create a tabular regression project
        regression_project = create_tabular_regression_project(
            client,
            name="Revenue Forecasting",
            description="Predict monthly revenue based on historical data",
        )

        # Example 4: Create a text classification project
        text_project = create_text_classification_project(
            client,
            name="Sentiment Analysis",
            description="Classify customer feedback sentiment",
        )

        # Example 5: List all projects
        list_all_projects(client)

        # Example 6: Filter projects by name
        list_projects_by_name(client, name="Customer")

        # Example 7: Filter projects by task type
        list_projects_by_task_type(client, task_type="llm-base")

        print("\n" + "=" * 70)
        print("✅ All examples completed successfully!")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()

