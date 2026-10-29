"""Deprecated compatibility imports for historical integrations."""

from app.backend.services.reflective_search_runner import (
    ReflectiveSearchEvent,
    ReflectiveSearchJob,
    find_new_output_dir,
    load_best_code_from_json,
    parse_lines,
    parse_log_line,
    prepare_reflective_search_runtime,
    start_reflective_search,
    tail_log,
)


EOHEvent = ReflectiveSearchEvent
EOHJob = ReflectiveSearchJob
prepare_eoh_runtime = prepare_reflective_search_runtime
start_eoh = start_reflective_search

__all__ = [
    "ReflectiveSearchEvent",
    "ReflectiveSearchJob",
    "find_new_output_dir",
    "load_best_code_from_json",
    "parse_lines",
    "parse_log_line",
    "prepare_reflective_search_runtime",
    "start_reflective_search",
    "tail_log",
    "EOHEvent",
    "EOHJob",
    "prepare_eoh_runtime",
    "start_eoh",
]
