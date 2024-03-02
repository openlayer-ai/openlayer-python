"""Basic Module that defines the version of the SDK.

   This module allows for the SDK version to be accessed from the SDK itself.
   See https://stackoverflow.com/questions/2058802

   Typical usage example:

      from .version import __version__

      CLIENT_METADATA = {"version": __version__}
      params = {
         "some_data": "some_value",
      }
      params.update(CLIENT_METADATA)
      res = https.request(
         method=method,
         url=url,
         headers=headers,
         params=params,
         json=body,
         files=files,
         data=data,
      )
"""

__version__ = "0.1.0a25"
