import os

_current_filepath = os.path.abspath(__file__)
_lib_directory = os.path.dirname(_current_filepath)
_src_directory = os.path.dirname(_lib_directory)

CODE_DIR = os.path.dirname(_src_directory)
RAW_DIR = os.path.join(CODE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(CODE_DIR, "data", "processed")
AGGREGATE_DIR = os.path.join(CODE_DIR, "data", "aggregate")
PHOTO_DIR = os.path.join(CODE_DIR, "data", "photos")
FIGURE_DIR = os.path.join(CODE_DIR, "data", "figures")
