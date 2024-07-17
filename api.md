# Projects

Types:

```python
from openlayer.types import ProjectCreateResponse, ProjectListResponse
```

Methods:

- <code title="post /projects">client.projects.<a href="./src/openlayer/resources/projects/projects.py">create</a>(\*\*<a href="src/openlayer/types/project_create_params.py">params</a>) -> <a href="./src/openlayer/types/project_create_response.py">ProjectCreateResponse</a></code>
- <code title="get /projects">client.projects.<a href="./src/openlayer/resources/projects/projects.py">list</a>(\*\*<a href="src/openlayer/types/project_list_params.py">params</a>) -> <a href="./src/openlayer/types/project_list_response.py">ProjectListResponse</a></code>

## Commits

Types:

```python
from openlayer.types.projects import CommitListResponse
```

Methods:

- <code title="get /projects/{projectId}/versions">client.projects.commits.<a href="./src/openlayer/resources/projects/commits.py">list</a>(project_id, \*\*<a href="src/openlayer/types/projects/commit_list_params.py">params</a>) -> <a href="./src/openlayer/types/projects/commit_list_response.py">CommitListResponse</a></code>

## InferencePipelines

Types:

```python
from openlayer.types.projects import InferencePipelineCreateResponse, InferencePipelineListResponse
```

Methods:

- <code title="post /projects/{projectId}/inference-pipelines">client.projects.inference_pipelines.<a href="./src/openlayer/resources/projects/inference_pipelines.py">create</a>(project_id, \*\*<a href="src/openlayer/types/projects/inference_pipeline_create_params.py">params</a>) -> <a href="./src/openlayer/types/projects/inference_pipeline_create_response.py">InferencePipelineCreateResponse</a></code>
- <code title="get /projects/{projectId}/inference-pipelines">client.projects.inference_pipelines.<a href="./src/openlayer/resources/projects/inference_pipelines.py">list</a>(project_id, \*\*<a href="src/openlayer/types/projects/inference_pipeline_list_params.py">params</a>) -> <a href="./src/openlayer/types/projects/inference_pipeline_list_response.py">InferencePipelineListResponse</a></code>

# Commits

## TestResults

Types:

```python
from openlayer.types.commits import TestResultListResponse
```

Methods:

- <code title="get /versions/{projectVersionId}/results">client.commits.test_results.<a href="./src/openlayer/resources/commits/test_results.py">list</a>(project_version_id, \*\*<a href="src/openlayer/types/commits/test_result_list_params.py">params</a>) -> <a href="./src/openlayer/types/commits/test_result_list_response.py">TestResultListResponse</a></code>

# InferencePipelines

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
