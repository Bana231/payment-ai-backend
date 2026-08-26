from pprint import pprint

from investigation_service import (
    run_investigation,
)


result = run_investigation(
    question=(
        "Why are payment transactions failing?"
    )
)


print("\n================================")
print("INVESTIGATION SERVICE RESULT")
print("================================\n")


pprint(
    result,
    sort_dicts=False,
)
