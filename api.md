# Projects

Types:

```python
from openlayer.types import ProjectCreateResponse, ProjectListResponse
```

Methods:

- <code title="post /projects">client.projects.<a href="./src/openlayer/resources/projects/projects.py">create</a>(\*\*<a href="src/openlayer/types/project_create_params.py">params</a>) -> <a href="./src/openlayer/types/project_create_response.py">ProjectCreateResponse</a></code>
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
from openlayer.types import InferencePipelineRetrieveResponse, InferencePipelineUpdateResponse
```

Methods:

- <code title="get /inference-pipelines/{inferencePipelineId}">client.inference_pipelines.<a href="./src/openlayer/resources/inference_pipelines/inference_pipelines.py">retrieve</a>(inference_pipeline_id, \*\*<a href="src/openlayer/types/inference_pipeline_retrieve_params.py">params</a>) -> <a href="./src/openlayer/types/inference_pipeline_retrieve_response.py">InferencePipelineRetrieveResponse</a></code>
- <code title="put /inference-pipelines/{inferencePipelineId}">client.inference_pipelines.<a href="./src/openlayer/resources/inference_pipelines/inference_pipelines.py">update</a>(inference_pipeline_id, \*\*<a href="src/openlayer/types/inference_pipeline_update_params.py">params</a>) -> <a href="./src/openlayer/types/inference_pipeline_update_response.py">InferencePipelineUpdateResponse</a></code>
- <code title="delete /inference-pipelines/{inferencePipelineId}">client.inference_pipelines.<a href="./src/openlayer/resources/inference_pipelines/inference_pipelines.py">delete</a>(inference_pipeline_id) -> None</code>

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
from openlayer.types.inference_pipelines import RowUpdateResponse
```

Methods:

- <code title="put /inference-pipelines/{inferencePipelineId}/rows">client.inference_pipelines.rows.<a href="./src/openlayer/resources/inference_pipelines/rows.py">update</a>(inference_pipeline_id, \*\*<a href="src/openlayer/types/inference_pipelines/row_update_params.py">params</a>) -> <a href="./src/openlayer/types/inference_pipelines/row_update_response.py">RowUpdateResponse</a></code>

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
from openlayer.types.storage import PresignedURLCreateResponse
```

Methods:

- <code title="post /storage/presigned-url">client.storage.presigned_url.<a href="./src/openlayer/resources/storage/presigned_url.py">create</a>(\*\*<a href="src/openlayer/types/storage/presigned_url_create_params.py">params</a>) -> <a href="./src/openlayer/types/storage/presigned_url_create_response.py">PresignedURLCreateResponse</a></code>

# Tests

Types:

```python
from openlayer.types import TestEvaluateResponse, TestListResultsResponse
```

Methods:

- <code title="post /tests/{testId}/evaluate">client.tests.<a href="./src/openlayer/resources/tests.py">evaluate</a>(test_id, \*\*<a href="src/openlayer/types/test_evaluate_params.py">params</a>) -> <a href="./src/openlayer/types/test_evaluate_response.py">TestEvaluateResponse</a></code>
- <code title="get /tests/{testId}/results">client.tests.<a href="./src/openlayer/resources/tests.py">list_results</a>(test_id, \*\*<a href="src/openlayer/types/test_list_results_params.py">params</a>) -> <a href="./src/openlayer/types/test_list_results_response.py">TestListResultsResponse</a></code>
