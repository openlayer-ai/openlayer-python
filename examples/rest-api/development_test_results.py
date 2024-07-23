import os

from openlayer import Openlayer

commit_id = "YOUR_OPENLAYER_COMMIT_ID"


client = Openlayer(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENLAYER_API_KEY"),
)
response = client.commits.test_results.list(commit_id=commit_id)

print(response.items)
