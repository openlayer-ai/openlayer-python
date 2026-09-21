# Projects

Types:

```python
from openlayer.types import ProjectCreateResponse, ProjectUpdateResponse, ProjectListResponse
```

Methods:

- <code title="post /projects">client.projects.<a href="./src/openlayer/resources/projects/projects.py">create</a>(\*\*<a href="src/openlayer/types/project_create_params.py">params</a>) -> <a href="./src/openlayer/types/project_create_response.py">ProjectCreateResponse</a></code>
- <code title="patch /projects/{projectId}">client.projects.<a href="./src/openlayer/resources/projects/projects.py">update</a>(project_id, \*\*<a href="src/openlayer/types/project_update_params.py">params</a>) -> <a href="./src/openlayer/types/project_update_response.py">ProjectUpdateResponse</a></code>
- <code title="get /projects">client.projects.<a href="./src/openlayer/resources/projects/projects.py">list</a>(\*\*<a href="src/openlayer/types/project_list_params.py">params</a>) -> <a href="./src/openlayer/types/project_list_response.py">ProjectListResponse</a></code>
- <code title="delete /projects/{projectId}">client.projects.<a href="./src/openlayer/resources/projects/projects.py">delete</a>(project_id) -> None</code>

## Commits

Types:

```python
from openlayer.types.projects import CommitCreateResponse, CommitListResponse
```

Methods:

- <code title="post /projects/{projectId}/versions">client.projects.commits.<a href="./src/openlayer/resources/projects/commits.py">create</a>(project_id, \*\*<a href="src/openlayer/types/projects/commit_create_params.py">params</a>) -> <a href="./src/openlayer/types/projects/commit_create_response.py">CommitCreateResponse</a></code>
- <code title="get /projects/{projectId}/versions">client.projects.commits.<a href="./src/openlayer/resources/projects/commits.py">list</a>(project_id, \*\*<a href="src/openlayer/types/projects/commit_list_params.py">params</a>) -> <a href="./src/openlayer/types/projects/commit_list_response.py">CommitListResponse</a></code>

## InferencePipelines

Types:

```python
from openlayer.types.projects import InferencePipelineCreateResponse, InferencePipelineListResponse
```

Methods:

- <code title="post /projects/{projectId}/inference-pipelines">client.projects.inference_pipelines.<a href="./src/openlayer/resources/projects/inference_pipelines.py">create</a>(project_id, \*\*<a href="src/openlayer/types/projects/inference_pipeline_create_params.py">params</a>) -> <a href="./src/openlayer/types/projects/inference_pipeline_create_response.py">InferencePipelineCreateResponse</a></code>
- <code title="get /projects/{projectId}/inference-pipelines">client.projects.inference_pipelines.<a href="./src/openlayer/resources/projects/inference_pipelines.py">list</a>(project_id, \*\*<a href="src/openlayer/types/projects/inference_pipeline_list_params.py">params</a>) -> <a href="./src/openlayer/types/projects/inference_pipeline_list_response.py">InferencePipelineListResponse</a></code>

## Tests

Types:

```python
from openlayer.types.projects import TestCreateResponse, TestUpdateResponse, TestListResponse
```

Methods:

- <code title="post /projects/{projectId}/tests">client.projects.tests.<a href="./src/openlayer/resources/projects/tests.py">create</a>(project_id, \*\*<a href="src/openlayer/types/projects/test_create_params.py">params</a>) -> <a href="./src/openlayer/types/projects/test_create_response.py">TestCreateResponse</a></code>
- <code title="put /projects/{projectId}/tests">client.projects.tests.<a href="./src/openlayer/resources/projects/tests.py">update</a>(project_id, \*\*<a href="src/openlayer/types/projects/test_update_params.py">params</a>) -> <a href="./src/openlayer/types/projects/test_update_response.py">TestUpdateResponse</a></code>
- <code title="get /projects/{projectId}/tests">client.projects.tests.<a href="./src/openlayer/resources/projects/tests.py">list</a>(project_id, \*\*<a href="src/openlayer/types/projects/test_list_params.py">params</a>) -> <a href="./src/openlayer/types/projects/test_list_response.py">TestListResponse</a></code>

# Workspaces

Types:

```python
from openlayer.types import WorkspaceRetrieveResponse, WorkspaceUpdateResponse
```

Methods:

- <code title="get /workspaces/{workspaceId}">client.workspaces.<a href="./src/openlayer/resources/workspaces/workspaces.py">retrieve</a>(workspace_id) -> <a href="./src/openlayer/types/workspace_retrieve_response.py">WorkspaceRetrieveResponse</a></code>
- <code title="put /workspaces/{workspaceId}">client.workspaces.<a href="./src/openlayer/resources/workspaces/workspaces.py">update</a>(workspace_id, \*\*<a href="src/openlayer/types/workspace_update_params.py">params</a>) -> <a href="./src/openlayer/types/workspace_update_response.py">WorkspaceUpdateResponse</a></code>

## Invites

Types:

```python
from openlayer.types.workspaces import InviteCreateResponse, InviteListResponse
```

Methods:

- <code title="post /workspaces/{workspaceId}/invites">client.workspaces.invites.<a href="./src/openlayer/resources/workspaces/invites.py">create</a>(workspace_id, \*\*<a href="src/openlayer/types/workspaces/invite_create_params.py">params</a>) -> <a href="./src/openlayer/types/workspaces/invite_create_response.py">InviteCreateResponse</a></code>
- <code title="get /workspaces/{workspaceId}/invites">client.workspaces.invites.<a href="./src/openlayer/resources/workspaces/invites.py">list</a>(workspace_id, \*\*<a href="src/openlayer/types/workspaces/invite_list_params.py">params</a>) -> <a href="./src/openlayer/types/workspaces/invite_list_response.py">InviteListResponse</a></code>

## APIKeys

Types:

```python
from openlayer.types.workspaces import APIKeyCreateResponse
```

Methods:

- <code title="post /workspaces/{workspaceId}/api-keys">client.workspaces.api_keys.<a href="./src/openlayer/resources/workspaces/api_keys.py">create</a>(workspace_id, \*\*<a href="src/openlayer/types/workspaces/api_key_create_params.py">params</a>) -> <a href="./src/openlayer/types/workspaces/api_key_create_response.py">APIKeyCreateResponse</a></code>

# Commits

Types:

```python
from openlayer.types import CommitRetrieveResponse
```

Methods:

- <code title="get /versions/{projectVersionId}">client.commits.<a href="./src/openlayer/resources/commits/commits.py">retrieve</a>(project_version_id) -> <a href="./src/openlayer/types/commit_retrieve_response.py">CommitRetrieveResponse</a></code>

## TestResults

Types:

```python
from openlayer.types.commits import TestResultListResponse
```

Methods:

- <code title="get /versions/{projectVersionId}/results">client.commits.test_results.<a href="./src/openlayer/resources/commits/test_results.py">list</a>(project_version_id, \*\*<a href="src/openlayer/types/commits/test_result_list_params.py">params</a>) -> <a href="./src/openlayer/types/commits/test_result_list_response.py">TestResultListResponse</a></code>

# InferencePipelines

Types:

```python
from openlayer.types import (
    InferencePipelineRetrieveResponse,
    InferencePipelineUpdateResponse,
    InferencePipelineRetrieveSessionsResponse,
    InferencePipelineRetrieveUsersResponse,
)
```

Methods:

- <code title="get /inference-pipelines/{inferencePipelineId}">client.inference_pipelines.<a href="./src/openlayer/resources/inference_pipelines/inference_pipelines.py">retrieve</a>(inference_pipeline_id, \*\*<a href="src/openlayer/types/inference_pipeline_retrieve_params.py">params</a>) -> <a href="./src/openlayer/types/inference_pipeline_retrieve_response.py">InferencePipelineRetrieveResponse</a></code>
- <code title="put /inference-pipelines/{inferencePipelineId}">client.inference_pipelines.<a href="./src/openlayer/resources/inference_pipelines/inference_pipelines.py">update</a>(inference_pipeline_id, \*\*<a href="src/openlayer/types/inference_pipeline_update_params.py">params</a>) -> <a href="./src/openlayer/types/inference_pipeline_update_response.py">InferencePipelineUpdateResponse</a></code>
- <code title="delete /inference-pipelines/{inferencePipelineId}">client.inference_pipelines.<a href="./src/openlayer/resources/inference_pipelines/inference_pipelines.py">delete</a>(inference_pipeline_id) -> None</code>
- <code title="post /inference-pipelines/{inferencePipelineId}/sessions">client.inference_pipelines.<a href="./src/openlayer/resources/inference_pipelines/inference_pipelines.py">retrieve_sessions</a>(inference_pipeline_id, \*\*<a href="src/openlayer/types/inference_pipeline_retrieve_sessions_params.py">params</a>) -> <a href="./src/openlayer/types/inference_pipeline_retrieve_sessions_response.py">InferencePipelineRetrieveSessionsResponse</a></code>
- <code title="post /inference-pipelines/{inferencePipelineId}/users">client.inference_pipelines.<a href="./src/openlayer/resources/inference_pipelines/inference_pipelines.py">retrieve_users</a>(inference_pipeline_id, \*\*<a href="src/openlayer/types/inference_pipeline_retrieve_users_params.py">params</a>) -> <a href="./src/openlayer/types/inference_pipeline_retrieve_users_response.py">InferencePipelineRetrieveUsersResponse</a></code>

## Data

Types:

```python
from openlayer.types.inference_pipelines import DataStreamResponse
```

Methods:

- <code title="post /inference-pipelines/{inferencePipelineId}/data-stream">client.inference_pipelines.data.<a href="./src/openlayer/resources/inference_pipelines/data.py">stream</a>(inference_pipeline_id, \*\*<a href="src/openlayer/types/inference_pipelines/data_stream_params.py">params</a>) -> <a href="./src/openlayer/types/inference_pipelines/data_stream_response.py">DataStreamResponse</a></code>

## Rows

Types:

```python
from openlayer.types.inference_pipelines import (
    RowRetrieveResponse,
    RowUpdateResponse,
    RowListResponse,
)
```

Methods:

- <code title="get /inference-pipelines/{inferencePipelineId}/rows/{inferenceId}">client.inference_pipelines.rows.<a href="./src/openlayer/resources/inference_pipelines/rows.py">retrieve</a>(inference_id, \*, inference_pipeline_id) -> <a href="./src/openlayer/types/inference_pipelines/row_retrieve_response.py">RowRetrieveResponse</a></code>
- <code title="put /inference-pipelines/{inferencePipelineId}/rows">client.inference_pipelines.rows.<a href="./src/openlayer/resources/inference_pipelines/rows.py">update</a>(inference_pipeline_id, \*\*<a href="src/openlayer/types/inference_pipelines/row_update_params.py">params</a>) -> <a href="./src/openlayer/types/inference_pipelines/row_update_response.py">RowUpdateResponse</a></code>
- <code title="post /inference-pipelines/{inferencePipelineId}/rows">client.inference_pipelines.rows.<a href="./src/openlayer/resources/inference_pipelines/rows.py">list</a>(inference_pipeline_id, \*\*<a href="src/openlayer/types/inference_pipelines/row_list_params.py">params</a>) -> <a href="./src/openlayer/types/inference_pipelines/row_list_response.py">RowListResponse</a></code>
- <code title="delete /inference-pipelines/{inferencePipelineId}/rows/{inferenceId}">client.inference_pipelines.rows.<a href="./src/openlayer/resources/inference_pipelines/rows.py">delete</a>(inference_id, \*, inference_pipeline_id) -> None</code>

## TestResults

Types:

```python
from openlayer.types.inference_pipelines import TestResultListResponse
```

Methods:

- <code title="get /inference-pipelines/{inferencePipelineId}/results">client.inference_pipelines.test_results.<a href="./src/openlayer/resources/inference_pipelines/test_results.py">list</a>(inference_pipeline_id, \*\*<a href="src/openlayer/types/inference_pipelines/test_result_list_params.py">params</a>) -> <a href="./src/openlayer/types/inference_pipelines/test_result_list_response.py">TestResultListResponse</a></code>

# Storage

## PresignedURL

Types:

```python
from openlayer.types.storage import PresignedURLCreateResponse, PresignedURLRetrieveResponse
```

Methods:

- <code title="post /storage/presigned-url">client.storage.presigned_url.<a href="./src/openlayer/resources/storage/presigned_url.py">create</a>(\*\*<a href="src/openlayer/types/storage/presigned_url_create_params.py">params</a>) -> <a href="./src/openlayer/types/storage/presigned_url_create_response.py">PresignedURLCreateResponse</a></code>
- <code title="get /storage/presigned-url">client.storage.presigned_url.<a href="./src/openlayer/resources/storage/presigned_url.py">retrieve</a>(\*\*<a href="src/openlayer/types/storage/presigned_url_retrieve_params.py">params</a>) -> <a href="./src/openlayer/types/storage/presigned_url_retrieve_response.py">PresignedURLRetrieveResponse</a></code>

# Tests

Types:

```python
from openlayer.types import TestEvaluateResponse, TestListResultsResponse
```

Methods:

- <code title="post /tests/{testId}/evaluate">client.tests.<a href="./src/openlayer/resources/tests.py">evaluate</a>(test_id, \*\*<a href="src/openlayer/types/test_evaluate_params.py">params</a>) -> <a href="./src/openlayer/types/test_evaluate_response.py">TestEvaluateResponse</a></code>
- <code title="get /tests/{testId}/results">client.tests.<a href="./src/openlayer/resources/tests.py">list_results</a>(test_id, \*\*<a href="src/openlayer/types/test_list_results_params.py">params</a>) -> <a href="./src/openlayer/types/test_list_results_response.py">TestListResultsResponse</a></code>

# BackgroundTasks

Types:

```python
from openlayer.types import BackgroundTaskRetrieveResponse
```

Methods:

- <code title="get /background-tasks/{taskId}">client.background_tasks.<a href="./src/openlayer/resources/background_tasks.py">retrieve</a>(task_id) -> <a href="./src/openlayer/types/background_task_retrieve_response.py">BackgroundTaskRetrieveResponse</a></code>

# Governance

## Frameworks

Types:

```python
from openlayer.types.governance import (
    FrameworkCreateResponse,
    FrameworkRetrieveResponse,
    FrameworkUpdateResponse,
    FrameworkListResponse,
    FrameworkExportResponse,
    FrameworkListProjectRuleStatsResponse,
    FrameworkListProjectsResponse,
    FrameworkListRulesResponse,
)
```

Methods:

- <code title="post /workspaces/{workspaceId}/frameworks">client.governance.frameworks.<a href="./src/openlayer/resources/governance/frameworks/frameworks.py">create</a>(workspace_id, \*\*<a href="src/openlayer/types/governance/framework_create_params.py">params</a>) -> <a href="./src/openlayer/types/governance/framework_create_response.py">FrameworkCreateResponse</a></code>
- <code title="get /frameworks/{frameworkId}">client.governance.frameworks.<a href="./src/openlayer/resources/governance/frameworks/frameworks.py">retrieve</a>(framework_id) -> <a href="./src/openlayer/types/governance/framework_retrieve_response.py">FrameworkRetrieveResponse</a></code>
- <code title="put /frameworks/{frameworkId}">client.governance.frameworks.<a href="./src/openlayer/resources/governance/frameworks/frameworks.py">update</a>(framework_id, \*\*<a href="src/openlayer/types/governance/framework_update_params.py">params</a>) -> <a href="./src/openlayer/types/governance/framework_update_response.py">FrameworkUpdateResponse</a></code>
- <code title="get /workspaces/{workspaceId}/frameworks">client.governance.frameworks.<a href="./src/openlayer/resources/governance/frameworks/frameworks.py">list</a>(workspace_id, \*\*<a href="src/openlayer/types/governance/framework_list_params.py">params</a>) -> <a href="./src/openlayer/types/governance/framework_list_response.py">FrameworkListResponse</a></code>
- <code title="post /frameworks/{frameworkId}/export">client.governance.frameworks.<a href="./src/openlayer/resources/governance/frameworks/frameworks.py">export</a>(framework_id, \*\*<a href="src/openlayer/types/governance/framework_export_params.py">params</a>) -> <a href="./src/openlayer/types/governance/framework_export_response.py">FrameworkExportResponse</a></code>
- <code title="get /frameworks/{frameworkId}/project-rule-stats">client.governance.frameworks.<a href="./src/openlayer/resources/governance/frameworks/frameworks.py">list_project_rule_stats</a>(framework_id, \*\*<a href="src/openlayer/types/governance/framework_list_project_rule_stats_params.py">params</a>) -> <a href="./src/openlayer/types/governance/framework_list_project_rule_stats_response.py">FrameworkListProjectRuleStatsResponse</a></code>
- <code title="get /frameworks/{frameworkId}/projects">client.governance.frameworks.<a href="./src/openlayer/resources/governance/frameworks/frameworks.py">list_projects</a>(framework_id, \*\*<a href="src/openlayer/types/governance/framework_list_projects_params.py">params</a>) -> <a href="./src/openlayer/types/governance/framework_list_projects_response.py">FrameworkListProjectsResponse</a></code>
- <code title="get /frameworks/{frameworkId}/rules">client.governance.frameworks.<a href="./src/openlayer/resources/governance/frameworks/frameworks.py">list_rules</a>(framework_id, \*\*<a href="src/openlayer/types/governance/framework_list_rules_params.py">params</a>) -> <a href="./src/openlayer/types/governance/framework_list_rules_response.py">FrameworkListRulesResponse</a></code>

### Documents

Types:

```python
from openlayer.types.governance.frameworks import DocumentRetrieveResponse, DocumentListResponse
```

Methods:

- <code title="get /frameworks/{frameworkId}/documents/{documentId}">client.governance.frameworks.documents.<a href="./src/openlayer/resources/governance/frameworks/documents.py">retrieve</a>(document_id, \*, framework_id) -> <a href="./src/openlayer/types/governance/frameworks/document_retrieve_response.py">DocumentRetrieveResponse</a></code>
- <code title="get /frameworks/{frameworkId}/documents">client.governance.frameworks.documents.<a href="./src/openlayer/resources/governance/frameworks/documents.py">list</a>(framework_id, \*\*<a href="src/openlayer/types/governance/frameworks/document_list_params.py">params</a>) -> <a href="./src/openlayer/types/governance/frameworks/document_list_response.py">DocumentListResponse</a></code>

### Sections

Types:

```python
from openlayer.types.governance.frameworks import SectionListRulesResponse
```

Methods:

- <code title="get /frameworks/{frameworkId}/sections/{sectionId}/rules">client.governance.frameworks.sections.<a href="./src/openlayer/resources/governance/frameworks/sections.py">list_rules</a>(section_id, \*, framework_id, \*\*<a href="src/openlayer/types/governance/frameworks/section_list_rules_params.py">params</a>) -> <a href="./src/openlayer/types/governance/frameworks/section_list_rules_response.py">SectionListRulesResponse</a></code>

### Subsections

Types:

```python
from openlayer.types.governance.frameworks import SubsectionListRulesResponse
```

Methods:

- <code title="get /frameworks/{frameworkId}/subsections/{subsectionId}/rules">client.governance.frameworks.subsections.<a href="./src/openlayer/resources/governance/frameworks/subsections.py">list_rules</a>(subsection_id, \*, framework_id, \*\*<a href="src/openlayer/types/governance/frameworks/subsection_list_rules_params.py">params</a>) -> <a href="./src/openlayer/types/governance/frameworks/subsection_list_rules_response.py">SubsectionListRulesResponse</a></code>

## Rules

Types:

```python
from openlayer.types.governance import (
    RuleCreateResponse,
    RuleRetrieveResponse,
    RuleUpdateResponse,
    RuleListResponse,
)
```

Methods:

- <code title="post /workspaces/{workspaceId}/rules">client.governance.rules.<a href="./src/openlayer/resources/governance/rules.py">create</a>(workspace_id, \*\*<a href="src/openlayer/types/governance/rule_create_params.py">params</a>) -> <a href="./src/openlayer/types/governance/rule_create_response.py">RuleCreateResponse</a></code>
- <code title="get /rules/{ruleId}">client.governance.rules.<a href="./src/openlayer/resources/governance/rules.py">retrieve</a>(rule_id) -> <a href="./src/openlayer/types/governance/rule_retrieve_response.py">RuleRetrieveResponse</a></code>
- <code title="put /rules/{ruleId}">client.governance.rules.<a href="./src/openlayer/resources/governance/rules.py">update</a>(rule_id, \*\*<a href="src/openlayer/types/governance/rule_update_params.py">params</a>) -> <a href="./src/openlayer/types/governance/rule_update_response.py">RuleUpdateResponse</a></code>
- <code title="get /workspaces/{workspaceId}/rules">client.governance.rules.<a href="./src/openlayer/resources/governance/rules.py">list</a>(workspace_id, \*\*<a href="src/openlayer/types/governance/rule_list_params.py">params</a>) -> <a href="./src/openlayer/types/governance/rule_list_response.py">RuleListResponse</a></code>
- <code title="delete /rules/{ruleId}">client.governance.rules.<a href="./src/openlayer/resources/governance/rules.py">delete</a>(rule_id) -> None</code>

## RuleResults

Types:

```python
from openlayer.types.governance import (
    RuleResultRetrieveResponse,
    RuleResultUpdateResponse,
    RuleResultListResponse,
    RuleResultCreateEvidenceResponse,
    RuleResultListEvidenceResponse,
)
```

Methods:

- <code title="get /rule-results/{ruleResultId}">client.governance.rule_results.<a href="./src/openlayer/resources/governance/rule_results.py">retrieve</a>(rule_result_id) -> <a href="./src/openlayer/types/governance/rule_result_retrieve_response.py">RuleResultRetrieveResponse</a></code>
- <code title="patch /rule-results/{ruleResultId}">client.governance.rule_results.<a href="./src/openlayer/resources/governance/rule_results.py">update</a>(rule_result_id, \*\*<a href="src/openlayer/types/governance/rule_result_update_params.py">params</a>) -> <a href="./src/openlayer/types/governance/rule_result_update_response.py">RuleResultUpdateResponse</a></code>
- <code title="get /workspaces/{workspaceId}/rule-results">client.governance.rule_results.<a href="./src/openlayer/resources/governance/rule_results.py">list</a>(workspace_id, \*\*<a href="src/openlayer/types/governance/rule_result_list_params.py">params</a>) -> <a href="./src/openlayer/types/governance/rule_result_list_response.py">RuleResultListResponse</a></code>
- <code title="post /rule-results/{ruleResultId}/evidence">client.governance.rule_results.<a href="./src/openlayer/resources/governance/rule_results.py">create_evidence</a>(rule_result_id, \*\*<a href="src/openlayer/types/governance/rule_result_create_evidence_params.py">params</a>) -> <a href="./src/openlayer/types/governance/rule_result_create_evidence_response.py">RuleResultCreateEvidenceResponse</a></code>
- <code title="get /rule-results/{ruleResultId}/evidence">client.governance.rule_results.<a href="./src/openlayer/resources/governance/rule_results.py">list_evidence</a>(rule_result_id, \*\*<a href="src/openlayer/types/governance/rule_result_list_evidence_params.py">params</a>) -> <a href="./src/openlayer/types/governance/rule_result_list_evidence_response.py">RuleResultListEvidenceResponse</a></code>

## RuleStats

Types:

```python
from openlayer.types.governance import RuleStatRetrieveResponse
```

Methods:

- <code title="get /workspaces/{workspaceId}/rule-stats">client.governance.rule_stats.<a href="./src/openlayer/resources/governance/rule_stats.py">retrieve</a>(workspace_id, \*\*<a href="src/openlayer/types/governance/rule_stat_retrieve_params.py">params</a>) -> <a href="./src/openlayer/types/governance/rule_stat_retrieve_response.py">RuleStatRetrieveResponse</a></code>

## RuleTags

Types:

```python
from openlayer.types.governance import RuleTagListResponse
```

Methods:

- <code title="get /workspaces/{workspaceId}/rule-tags">client.governance.rule_tags.<a href="./src/openlayer/resources/governance/rule_tags.py">list</a>(workspace_id, \*\*<a href="src/openlayer/types/governance/rule_tag_list_params.py">params</a>) -> <a href="./src/openlayer/types/governance/rule_tag_list_response.py">RuleTagListResponse</a></code>
