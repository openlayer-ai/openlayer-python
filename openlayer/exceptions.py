"""A collection of the different Openlayer Python client exceptions and their error codes.

Typical usage example:

  if project is None:
    raise errors.OpenlayerResourceNotFound(f"Project {project_id} does not exist")
"""
from typing import Dict


class OpenlayerException(Exception):
    """Generic OpenlayerException class"""

    code = None

    def __init__(self, message, errcode=None):
        if not message:
            message = type(self).__name__
        self.message = message

        if errcode:
            self.code = errcode

        if self.code:
            super().__init__(f"<Response [{self.code}]> {message}")
        else:
            super().__init__(f"<Response> {message}")


class OpenlayerValidationError(OpenlayerException):
    """Failed resource validations"""

    def __init__(self, message):
        super().__init__(message)


class OpenlayerSubscriptionPlanException(OpenlayerException):
    """Subscription plan exception class"""

    def __init__(self, message, context=None, mitigation=None):
        context = context or "You have reached your subscription plan's limits. \n"
        mitigation = mitigation or "To upgrade your plan, visit https://openlayer.com"
        super().__init__(context + message + mitigation)


class OpenlayerInvalidRequest(OpenlayerException):
    """400 - Bad Request -- The request was unacceptable,
    often due to missing a required parameter.
    """

    code = 400


class OpenlayerUnauthorized(OpenlayerException):
    """401 - Unauthorized -- No valid API key provided."""

    code = 401


class OpenlayerNotEnabled(OpenlayerException):
    """402 - Not enabled -- Please contact sales@openlayer.com before
    creating this type of task.
    """

    code = 402


class OpenlayerResourceNotFound(OpenlayerException):
    """404 - Not Found -- The requested resource doesn't exist."""

    code = 404


class OpenlayerDuplicateTask(OpenlayerException):
    """409 - Conflict -- The provided idempotency key or unique_id is
    already in use for a different request.
    """

    code = 409


class OpenlayerTooManyRequests(OpenlayerException):
    """429 - Too Many Requests -- Too many requests hit the API
    too quickly.
    """

    code = 429


class OpenlayerInternalError(OpenlayerException):
    """500 - Internal Server Error -- We had a problem with our server.
    Try again later.
    """

    code = 500


class OpenlayerServiceUnavailable(OpenlayerException):
    """503 - Server Timeout From Request Queueing -- Try again later."""

    code = 503


class OpenlayerTimeoutError(OpenlayerException):
    """504 - Server Timeout Error -- Try again later."""

    code = 504


ExceptionMap: Dict[int, OpenlayerException] = {
    OpenlayerInvalidRequest.code: OpenlayerInvalidRequest,
    OpenlayerUnauthorized.code: OpenlayerUnauthorized,
    OpenlayerNotEnabled.code: OpenlayerNotEnabled,
    OpenlayerResourceNotFound.code: OpenlayerResourceNotFound,
    OpenlayerDuplicateTask.code: OpenlayerDuplicateTask,
    OpenlayerTooManyRequests.code: OpenlayerTooManyRequests,
    OpenlayerInternalError.code: OpenlayerInternalError,
    OpenlayerTimeoutError.code: OpenlayerTimeoutError,
    OpenlayerServiceUnavailable.code: OpenlayerServiceUnavailable,
}
