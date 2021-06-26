from typing import Dict


class UnboxException(Exception):
    """Generic UnboxException class"""

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


class UnboxInvalidRequest(UnboxException):
    """400 - Bad Request -- The request was unacceptable,
    often due to missing a required parameter.
    """

    code = 400


class UnboxUnauthorized(UnboxException):
    """401 - Unauthorized -- No valid API key provided. """

    code = 401


class UnboxNotEnabled(UnboxException):
    """402 - Not enabled -- Please contact sales@tryunbox.ai before
    creating this type of task.
    """

    code = 402


class UnboxResourceNotFound(UnboxException):
    """404 - Not Found -- The requested resource doesn't exist. """

    code = 404


class UnboxDuplicateTask(UnboxException):
    """409 - Conflict -- The provided idempotency key or unique_id is
    already in use for a different request.
    """

    code = 409


class UnboxTooManyRequests(UnboxException):
    """429 - Too Many Requests -- Too many requests hit the API
    too quickly.
    """

    code = 429


class UnboxInternalError(UnboxException):
    """500 - Internal Server Error -- We had a problem with our server.
    Try again later.
    """

    code = 500


class UnboxServiceUnavailable(UnboxException):
    """503 - Server Timeout From Request Queueing -- Try again later. """

    code = 503


class UnboxTimeoutError(UnboxException):
    """504 - Server Timeout Error -- Try again later. """

    code = 504


ExceptionMap: Dict[int, UnboxException] = {
    UnboxInvalidRequest.code: UnboxInvalidRequest,
    UnboxUnauthorized.code: UnboxUnauthorized,
    UnboxNotEnabled.code: UnboxNotEnabled,
    UnboxResourceNotFound.code: UnboxResourceNotFound,
    UnboxDuplicateTask.code: UnboxDuplicateTask,
    UnboxTooManyRequests.code: UnboxTooManyRequests,
    UnboxInternalError.code: UnboxInternalError,
    UnboxTimeoutError.code: UnboxTimeoutError,
    UnboxServiceUnavailable.code: UnboxServiceUnavailable,
}
