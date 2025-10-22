# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import List, Union, Optional
from datetime import date, datetime
from typing_extensions import Literal, TypeAlias

from pydantic import Field as FieldInfo

from ..._models import BaseModel

__all__ = [
    "InferencePipelineListResponse",
    "Item",
    "ItemLinks",
    "ItemDataBackend",
    "ItemDataBackendUnionMember0",
    "ItemDataBackendUnionMember0Config",
    "ItemDataBackendBackendType",
    "ItemDataBackendUnionMember2",
    "ItemDataBackendUnionMember2Config",
    "ItemDataBackendUnionMember3",
    "ItemDataBackendUnionMember3Config",
    "ItemDataBackendUnionMember4",
    "ItemDataBackendUnionMember4Config",
    "ItemDataBackendUnionMember5",
    "ItemDataBackendUnionMember5Config",
    "ItemProject",
    "ItemProjectLinks",
    "ItemProjectGitRepo",
    "ItemWorkspace",
    "ItemWorkspaceMonthlyUsage",
]


class ItemLinks(BaseModel):
    app: str


class ItemDataBackendUnionMember0Config(BaseModel):
    ground_truth_column_name: Optional[str] = FieldInfo(alias="groundTruthColumnName", default=None)
    """Name of the column with the ground truths."""

    human_feedback_column_name: Optional[str] = FieldInfo(alias="humanFeedbackColumnName", default=None)
    """Name of the column with human feedback."""

    latency_column_name: Optional[str] = FieldInfo(alias="latencyColumnName", default=None)
    """Name of the column with the latencies."""

    timestamp_column_name: Optional[str] = FieldInfo(alias="timestampColumnName", default=None)
    """Name of the column with the timestamps.

    Timestamps must be in UNIX sec format. If not provided, the upload timestamp is
    used.
    """


class ItemDataBackendUnionMember0(BaseModel):
    backend_type: Literal["bigquery"] = FieldInfo(alias="backendType")

    bigquery_connection_id: Optional[str] = FieldInfo(alias="bigqueryConnectionId", default=None)

    dataset_id: str = FieldInfo(alias="datasetId")

    project_id: str = FieldInfo(alias="projectId")

    table_id: Optional[str] = FieldInfo(alias="tableId", default=None)

    partition_type: Optional[Literal["DAY", "MONTH", "YEAR"]] = FieldInfo(alias="partitionType", default=None)


class ItemDataBackendBackendType(BaseModel):
    backend_type: Literal["default"] = FieldInfo(alias="backendType")


class ItemDataBackendUnionMember2Config(BaseModel):
    ground_truth_column_name: Optional[str] = FieldInfo(alias="groundTruthColumnName", default=None)
    """Name of the column with the ground truths."""

    human_feedback_column_name: Optional[str] = FieldInfo(alias="humanFeedbackColumnName", default=None)
    """Name of the column with human feedback."""

    latency_column_name: Optional[str] = FieldInfo(alias="latencyColumnName", default=None)
    """Name of the column with the latencies."""

    timestamp_column_name: Optional[str] = FieldInfo(alias="timestampColumnName", default=None)
    """Name of the column with the timestamps.

    Timestamps must be in UNIX sec format. If not provided, the upload timestamp is
    used.
    """


class ItemDataBackendUnionMember2(BaseModel):
    backend_type: Literal["snowflake"] = FieldInfo(alias="backendType")

    database: str

    schema_: str = FieldInfo(alias="schema")

    snowflake_connection_id: Optional[str] = FieldInfo(alias="snowflakeConnectionId", default=None)

    table: Optional[str] = None


class ItemDataBackendUnionMember3Config(BaseModel):
    ground_truth_column_name: Optional[str] = FieldInfo(alias="groundTruthColumnName", default=None)
    """Name of the column with the ground truths."""

    human_feedback_column_name: Optional[str] = FieldInfo(alias="humanFeedbackColumnName", default=None)
    """Name of the column with human feedback."""

    latency_column_name: Optional[str] = FieldInfo(alias="latencyColumnName", default=None)
    """Name of the column with the latencies."""

    timestamp_column_name: Optional[str] = FieldInfo(alias="timestampColumnName", default=None)
    """Name of the column with the timestamps.

    Timestamps must be in UNIX sec format. If not provided, the upload timestamp is
    used.
    """


class ItemDataBackendUnionMember3(BaseModel):
    backend_type: Literal["databricks_dtl"] = FieldInfo(alias="backendType")

    databricks_dtl_connection_id: Optional[str] = FieldInfo(alias="databricksDtlConnectionId", default=None)

    table_id: Optional[str] = FieldInfo(alias="tableId", default=None)


class ItemDataBackendUnionMember4Config(BaseModel):
    ground_truth_column_name: Optional[str] = FieldInfo(alias="groundTruthColumnName", default=None)
    """Name of the column with the ground truths."""

    human_feedback_column_name: Optional[str] = FieldInfo(alias="humanFeedbackColumnName", default=None)
    """Name of the column with human feedback."""

    latency_column_name: Optional[str] = FieldInfo(alias="latencyColumnName", default=None)
    """Name of the column with the latencies."""

    timestamp_column_name: Optional[str] = FieldInfo(alias="timestampColumnName", default=None)
    """Name of the column with the timestamps.

    Timestamps must be in UNIX sec format. If not provided, the upload timestamp is
    used.
    """


class ItemDataBackendUnionMember4(BaseModel):
    backend_type: Literal["redshift"] = FieldInfo(alias="backendType")

    redshift_connection_id: Optional[str] = FieldInfo(alias="redshiftConnectionId", default=None)

    schema_name: str = FieldInfo(alias="schemaName")

    table_name: str = FieldInfo(alias="tableName")


class ItemDataBackendUnionMember5Config(BaseModel):
    ground_truth_column_name: Optional[str] = FieldInfo(alias="groundTruthColumnName", default=None)
    """Name of the column with the ground truths."""

    human_feedback_column_name: Optional[str] = FieldInfo(alias="humanFeedbackColumnName", default=None)
    """Name of the column with human feedback."""

    latency_column_name: Optional[str] = FieldInfo(alias="latencyColumnName", default=None)
    """Name of the column with the latencies."""

    timestamp_column_name: Optional[str] = FieldInfo(alias="timestampColumnName", default=None)
    """Name of the column with the timestamps.

    Timestamps must be in UNIX sec format. If not provided, the upload timestamp is
    used.
    """


class ItemDataBackendUnionMember5(BaseModel):
    backend_type: Literal["postgres"] = FieldInfo(alias="backendType")

    database: str

    postgres_connection_id: Optional[str] = FieldInfo(alias="postgresConnectionId", default=None)

    schema_: str = FieldInfo(alias="schema")

    table: Optional[str] = None


ItemDataBackend: TypeAlias = Union[
    ItemDataBackendUnionMember0,
    ItemDataBackendBackendType,
    ItemDataBackendUnionMember2,
    ItemDataBackendUnionMember3,
    ItemDataBackendUnionMember4,
    ItemDataBackendUnionMember5,
    None,
]


class ItemProjectLinks(BaseModel):
    app: str


class ItemProjectGitRepo(BaseModel):
    id: str

    date_connected: datetime = FieldInfo(alias="dateConnected")

    date_updated: datetime = FieldInfo(alias="dateUpdated")

    git_account_id: str = FieldInfo(alias="gitAccountId")

    git_id: int = FieldInfo(alias="gitId")

    name: str

    private: bool

    project_id: str = FieldInfo(alias="projectId")

    slug: str

    url: str

    branch: Optional[str] = None

    root_dir: Optional[str] = FieldInfo(alias="rootDir", default=None)


class ItemProject(BaseModel):
    id: str
    """The project id."""

    creator_id: Optional[str] = FieldInfo(alias="creatorId", default=None)
    """The project creator id."""

    date_created: datetime = FieldInfo(alias="dateCreated")
    """The project creation date."""

    date_updated: datetime = FieldInfo(alias="dateUpdated")
    """The project last updated date."""

    development_goal_count: int = FieldInfo(alias="developmentGoalCount")
    """The number of tests in the development mode of the project."""

    goal_count: int = FieldInfo(alias="goalCount")
    """The total number of tests in the project."""

    inference_pipeline_count: int = FieldInfo(alias="inferencePipelineCount")
    """The number of inference pipelines in the project."""

    links: ItemProjectLinks
    """Links to the project."""

    monitoring_goal_count: int = FieldInfo(alias="monitoringGoalCount")
    """The number of tests in the monitoring mode of the project."""

    name: str
    """The project name."""

    source: Optional[Literal["web", "api", "null"]] = None
    """The source of the project."""

    task_type: Literal["llm-base", "tabular-classification", "tabular-regression", "text-classification"] = FieldInfo(
        alias="taskType"
    )
    """The task type of the project."""

    version_count: int = FieldInfo(alias="versionCount")
    """The number of versions (commits) in the project."""

    workspace_id: Optional[str] = FieldInfo(alias="workspaceId", default=None)
    """The workspace id."""

    description: Optional[str] = None
    """The project description."""

    git_repo: Optional[ItemProjectGitRepo] = FieldInfo(alias="gitRepo", default=None)


class ItemWorkspaceMonthlyUsage(BaseModel):
    execution_time_ms: Optional[int] = FieldInfo(alias="executionTimeMs", default=None)

    month_year: Optional[date] = FieldInfo(alias="monthYear", default=None)

    prediction_count: Optional[int] = FieldInfo(alias="predictionCount", default=None)


class ItemWorkspace(BaseModel):
    id: str
    """The workspace id."""

    creator_id: Optional[str] = FieldInfo(alias="creatorId", default=None)
    """The workspace creator id."""

    date_created: datetime = FieldInfo(alias="dateCreated")
    """The workspace creation date."""

    date_updated: datetime = FieldInfo(alias="dateUpdated")
    """The workspace last updated date."""

    invite_count: int = FieldInfo(alias="inviteCount")
    """The number of invites in the workspace."""

    member_count: int = FieldInfo(alias="memberCount")
    """The number of members in the workspace."""

    name: str
    """The workspace name."""

    period_end_date: Optional[datetime] = FieldInfo(alias="periodEndDate", default=None)
    """The end date of the current billing period."""

    period_start_date: Optional[datetime] = FieldInfo(alias="periodStartDate", default=None)
    """The start date of the current billing period."""

    project_count: int = FieldInfo(alias="projectCount")
    """The number of projects in the workspace."""

    slug: str
    """The workspace slug."""

    status: Literal[
        "active", "past_due", "unpaid", "canceled", "incomplete", "incomplete_expired", "trialing", "paused"
    ]

    monthly_usage: Optional[List[ItemWorkspaceMonthlyUsage]] = FieldInfo(alias="monthlyUsage", default=None)

    saml_only_access: Optional[bool] = FieldInfo(alias="samlOnlyAccess", default=None)
    """Whether the workspace only allows SAML authentication."""

    wildcard_domains: Optional[List[str]] = FieldInfo(alias="wildcardDomains", default=None)


class Item(BaseModel):
    id: str
    """The inference pipeline id."""

    date_created: datetime = FieldInfo(alias="dateCreated")
    """The creation date."""

    date_last_evaluated: Optional[datetime] = FieldInfo(alias="dateLastEvaluated", default=None)
    """The last test evaluation date."""

    date_last_sample_received: Optional[datetime] = FieldInfo(alias="dateLastSampleReceived", default=None)
    """The last data sample received date."""

    date_of_next_evaluation: Optional[datetime] = FieldInfo(alias="dateOfNextEvaluation", default=None)
    """The next test evaluation date."""

    date_updated: datetime = FieldInfo(alias="dateUpdated")
    """The last updated date."""

    description: Optional[str] = None
    """The inference pipeline description."""

    failing_goal_count: int = FieldInfo(alias="failingGoalCount")
    """The number of tests failing."""

    links: ItemLinks

    name: str
    """The inference pipeline name."""

    passing_goal_count: int = FieldInfo(alias="passingGoalCount")
    """The number of tests passing."""

    project_id: str = FieldInfo(alias="projectId")
    """The project id."""

    status: Literal["queued", "running", "paused", "failed", "completed", "unknown"]
    """The status of test evaluation for the inference pipeline."""

    status_message: Optional[str] = FieldInfo(alias="statusMessage", default=None)
    """The status message of test evaluation for the inference pipeline."""

    total_goal_count: int = FieldInfo(alias="totalGoalCount")
    """The total number of tests."""

    data_backend: Optional[ItemDataBackend] = FieldInfo(alias="dataBackend", default=None)

    date_last_polled: Optional[datetime] = FieldInfo(alias="dateLastPolled", default=None)
    """The last time the data was polled."""

    project: Optional[ItemProject] = None

    total_records_count: Optional[int] = FieldInfo(alias="totalRecordsCount", default=None)
    """The total number of records in the data backend."""

    workspace: Optional[ItemWorkspace] = None

    workspace_id: Optional[str] = FieldInfo(alias="workspaceId", default=None)
    """The workspace id."""


class InferencePipelineListResponse(BaseModel):
    items: List[Item]
