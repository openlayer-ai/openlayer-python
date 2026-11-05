"""Keep dataset registration behind a successful storage upload."""

from __future__ import annotations

import json
import socket
from typing import Any, Protocol
from dataclasses import field, dataclass

import httpx
import pandas as pd  # pyright: ignore[reportMissingTypeStubs]
import pytest
import requests

from openlayer import Openlayer
from openlayer.lib import data
from openlayer.types.inference_pipelines import data_stream_params


# ----------------------------- Helper functions ----------------------------- #
@dataclass
class Boundary:
    status: int
    events: list[str] = field(default_factory=lambda: list[str]())
    registrations: list[dict[str, Any]] = field(default_factory=lambda: list[dict[str, Any]]())
    transport_error: bool = False

    def api(self, request: httpx.Request) -> httpx.Response:
        """Keep generated request construction and response parsing in the path."""
        assert request.url.host == "api.openlayer.invalid"
        if request.method == "POST" and request.url.path == "/storage/presigned-url":
            self.events.append("presign")
            object_name = request.url.params["objectName"]
            return httpx.Response(
                200,
                json={
                    "url": "https://storage.openlayer.invalid/upload",
                    "storageUri": "s3://synthetic-bucket/" + object_name,
                    "fields": {"key": object_name, "policy": "synthetic-policy"},
                },
            )
        assert (request.method, request.url.path) in {
            ("PUT", "/inference-pipelines/pipeline-synthetic"),
            ("POST", "/inference-pipelines/pipeline-synthetic/data"),
        }
        self.events.append("register")
        self.registrations.append(json.loads(request.content))
        return httpx.Response(200, json={"id": "pipeline-synthetic"})

    def storage(self, request: requests.PreparedRequest) -> requests.Response:
        """Replace the external storage service after request preparation."""
        assert request.url == "https://storage.openlayer.invalid/upload"
        self.events.append("storage")
        if self.transport_error:
            raise requests.Timeout("synthetic storage timeout")
        response = requests.Response()
        response.status_code = self.status
        response._content = b"synthetic storage response"
        response.request = request
        response.url = request.url
        return response


class BoundaryFactory(Protocol):
    def __call__(self, status: int = 204, transport_error: bool = False) -> Boundary: ...


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Turn a missing mock into a local failure before it can open a connection."""

    def deny(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Live network access is forbidden")

    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket, "create_connection", deny)


@pytest.fixture
def boundary_factory(monkeypatch: pytest.MonkeyPatch) -> BoundaryFactory:
    def create(status: int = 204, transport_error: bool = False) -> Boundary:
        boundary = Boundary(status=status, transport_error=transport_error)

        def send(_session: requests.Session, request: requests.PreparedRequest, **_kwargs: object) -> requests.Response:
            return boundary.storage(request)

        monkeypatch.setattr(requests.sessions.Session, "send", send)
        return boundary

    return create


def invoke(boundary: Boundary, storage: data.StorageType, workflow: str) -> None:
    """Use a real client so a failed upload can be observed at registration."""
    frame = pd.DataFrame({"feature": [10, 20], "prediction": [0, 1]})
    config: data_stream_params.Config = {
        "class_names": ["negative", "positive"],
        "feature_names": ["feature"],
        "predictions_column_name": "prediction",
    }
    with httpx.Client(transport=httpx.MockTransport(boundary.api), trust_env=False) as http:
        with Openlayer(
            api_key="synthetic-not-a-credential",
            base_url="https://api.openlayer.invalid",
            http_client=http,
            max_retries=0,
        ) as client:
            upload = data.upload_reference_dataframe if workflow == "reference" else data.upload_batch_inferences
            upload(
                client=client,
                inference_pipeline_id="pipeline-synthetic",
                dataset_df=frame,
                config=config,
                storage_type=storage,
            )


# --------------------------- End helper functions --------------------------- #


@pytest.mark.parametrize("storage", list(data.StorageType))
@pytest.mark.parametrize("status", [200, 201, 204])
def test_reference_success_registers_after_upload(
    boundary_factory: BoundaryFactory, storage: data.StorageType, status: int
) -> None:
    boundary = boundary_factory(status)
    invoke(boundary, storage, "reference")
    assert boundary.events == ["presign", "storage", "register"]
    assert len(boundary.registrations) == 1
    assert set(boundary.registrations[0]) == {"referenceDatasetUri"}
    assert boundary.registrations[0]["referenceDatasetUri"].startswith("s3://synthetic-bucket/reference_dataset_")


@pytest.mark.parametrize("storage", list(data.StorageType))
@pytest.mark.parametrize("workflow", ["reference", "batch"])
@pytest.mark.parametrize("status", [403, 503])
def test_rejected_storage_does_not_register(
    boundary_factory: BoundaryFactory, storage: data.StorageType, workflow: str, status: int
) -> None:
    """Rejecting the object must prevent publication of its storage URI."""
    boundary = boundary_factory(status)
    with pytest.raises(ValueError, match="Failed to upload file to storage: synthetic storage response"):
        invoke(boundary, storage, workflow)
    assert boundary.events == ["presign", "storage"]
    assert boundary.registrations == []


@pytest.mark.parametrize("storage", list(data.StorageType))
@pytest.mark.parametrize("workflow", ["reference", "batch"])
def test_transport_failure_does_not_register(
    boundary_factory: BoundaryFactory, storage: data.StorageType, workflow: str
) -> None:
    boundary = boundary_factory(transport_error=True)
    with pytest.raises(requests.Timeout, match="synthetic storage timeout"):
        invoke(boundary, storage, workflow)
    assert boundary.events == ["presign", "storage"]
    assert boundary.registrations == []


@pytest.mark.parametrize("storage", list(data.StorageType))
def test_batch_success_control(boundary_factory: BoundaryFactory, storage: data.StorageType) -> None:
    boundary = boundary_factory()
    invoke(boundary, storage, "batch")
    assert boundary.events == ["presign", "storage", "register"]
    assert len(boundary.registrations) == 1
    registration = boundary.registrations[0]
    assert registration["performDataMerge"] is False
    assert registration["storageUri"].endswith(".arrow")
    assert registration["config"] == {
        "classNames": ["negative", "positive"],
        "featureNames": ["feature"],
        "predictionsColumnName": "prediction",
    }
