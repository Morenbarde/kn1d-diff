# Shared function definition to simplify print statements done under the debrief options

import json
from typing import Any

#Load json file
def get_config(file_path:str) -> dict[str, Any]:
    with open(file_path, 'r') as config:
        return json.load(config)


def debrief(statement, condition):
    if condition:
        print(statement)