"""Deprecated compatibility entry point for historical configurations.

New code should import :class:`ReflectiveMetricSearch` from
``reflective_metric_search``.  These aliases keep old experiments runnable
without presenting the legacy label as BATTWISE's method name.
"""

from reflective_metric_search import (
    HierarchicalReflectionMemory,
    ReflectiveMetricSearch,
)


ReflectionLibrary = HierarchicalReflectionMemory
EOH_Textgrad = ReflectiveMetricSearch

__all__ = [
    "HierarchicalReflectionMemory",
    "ReflectiveMetricSearch",
    "ReflectionLibrary",
    "EOH_Textgrad",
]
