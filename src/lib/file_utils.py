import os
from src.lib.globals import *
import pandas as pd
from typing import List


def get_files(directory: str) -> List[str]:
    assert os.path.exists(directory)

    found_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.startswith('.'):
                continue
            relative_filepath = os.path.join(root, file)
            absolute_filepath = os.path.abspath(relative_filepath)
            found_files.append(absolute_filepath)

    if len(found_files) == 0:
        print(f"\tFound no files in {directory}")

    return found_files


def files_search(*terms: str, directory: str=CODE_DIR) -> List[str]:
    assert os.path.exists(directory)

    files = get_files(directory)
    terms = [
        word.lower()
        for string in terms
        for word in string.split()
    ]

    def matches_terms(filepath: str) -> bool:
        filename = os.path.basename(filepath)
        return (all(
            term.lower() in filename.lower()
            for term in terms
        ))
    
    filtered_files = list(filter(matches_terms, files))
    return filtered_files


