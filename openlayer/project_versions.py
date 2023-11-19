"""Module for the ProjectVersion class."""

import enum
import time
from typing import Optional

import tabulate


class TaskStatus(enum.Enum):
    """An enum containing the possible states of a project version."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    QUEUED = "queued"
    PAUSED = "paused"
    UNKNOWN = "unknown"


class ProjectVersion:
    """An object containing information about a project version on the
    Openlayer platform.

    This object is returned by the :meth:`openlayer.OpenlayerClient.push` and
    :meth:`openlayer.OpenlayerClient.load_project_version` methods.

    Refer to :meth:`openlayer.OpenlayerClient.load_project_version` for an example
    of how to use the object.
    """

    def __init__(self, json, client):
        self._json = json
        self.id = json["id"]
        self.client = client

    def __getattr__(self, name):
        if name in self._json:
            return self._json[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute {name}")

    def __hash__(self):
        return hash(self.id)

    def __str__(self):
        return f"ProjectVersion(id={self.id})"

    def __repr__(self):
        return f"ProjectVersion({self._json})"

    def to_dict(self):
        """Returns object properties as a dict.

        Returns
        -------
        Dict with object properties.
        """
        return self._json

    @property
    def status(self) -> TaskStatus:
        """Returns the current state of the project version."""
        return TaskStatus(self._json["status"])

    @property
    def status_message(self) -> str:
        """Returns the status message of the project version."""
        return self._json["statusMessage"]

    @property
    def passing_test_count(self) -> int:
        """Returns the number of passing tests for the project version."""
        return self._json["passingGoalCount"]

    @property
    def failing_test_count(self) -> int:
        """Returns the number of failing tests for the project version."""
        return self._json["failingGoalCount"]

    @property
    def skipped_test_count(self) -> int:
        """Returns the number of failing tests for the project version."""
        return (
            self._json["totalGoalCount"]
            - self._json["passingGoalCount"]
            - self._json["failingGoalCount"]
        )

    @property
    def total_test_count(self) -> int:
        """Returns the number of failing tests for the project version."""
        return self._json["totalGoalCount"]

    def wait_for_completion(self, timeout: Optional[int] = None):
        """Waits for the project version to complete.

        Parameters
        ----------
        timeout : int, optional
            Number of seconds to wait before timing out. If None, waits
            indefinitely.

        Returns
        -------
        ProjectVersion
            The project version object.
        """
        self.print_status_report()
        while self.status not in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
            prev_status_msg = self.status_message
            self.refresh()
            if self.status_message != prev_status_msg:
                self.print_status_report()
            time.sleep(1)
            if timeout:
                timeout -= 1
                if timeout <= 0:
                    print(
                        "Timeout exceeded. Visit the Openlayer dashboard to"
                        " check the status of the project version."
                    )
                    break
        if self.status == TaskStatus.FAILED:
            print("Project version failed with message:", self.status_message)
        elif self.status == TaskStatus.COMPLETED:
            print("Project version processed successfully.")

    def refresh(self):
        """Refreshes the project version object with the latest
        information from the server."""
        self._json = self.client.load_project_version(self.id).to_dict()

    def print_status_report(self):
        """Prints the status report along with its status message."""
        print("Status:", self.status.value, "(" + f"{self.status_message}" + ")")

    def print_test_report(self):
        """Prints the test results of the project version."""
        if self.status != TaskStatus.COMPLETED:
            print("Project version is not complete. Nothing to print.")
            return
        print(
            tabulate.tabulate(
                [
                    ["Passed", self.passing_test_count],
                    ["Failed", self.failing_test_count],
                    ["Skipped", self.skipped_test_count],
                    ["Total", self.total_test_count],
                ],
                headers=["Tests", "Count"],
                tablefmt="fancy_grid",
            ),
            f"\nVisit {self.links['app']} to view detailed results.",
        )
