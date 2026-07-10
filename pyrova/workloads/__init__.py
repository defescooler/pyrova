"""Workload (power-scenario) generators."""
from .structured import StructuredWorkloadModel, CorrelatedWorkloadModel
from .real_traces import RealTraceWorkloadModel
from .boom_traces import BoomWorkload

__all__ = ["StructuredWorkloadModel", "CorrelatedWorkloadModel",
           "RealTraceWorkloadModel", "BoomWorkload"]
