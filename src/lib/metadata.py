import os


PROTOCOL = 0
SUBJECT = 1
TRIAL = 2
EXPRESSION = 3


def _get_components(filename: str) -> str:
    basename = os.path.basename(filename)
    components = basename.split('_')
    return components


def get_protocol(filename: str) -> str:
    return _get_components(filename)[PROTOCOL]


def get_subject(filename: str) -> str:
    return _get_components(filename)[SUBJECT]


def get_expression(filename: str) -> str:
    return _get_components(filename)[EXPRESSION]


def get_trial(filename: str) -> str:
    return _get_components(filename)[TRIAL]
