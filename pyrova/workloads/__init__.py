"""Workload (power-scenario) generators."""
from .structured import StructuredWorkloadModel
from .real_traces import RealTraceWorkloadModel

__all__ = ["StructuredWorkloadModel", "RealTraceWorkloadModel"]
