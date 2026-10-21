"""Metrics and optional video preprocessing utilities."""
from .metrics import SkillMetrics


def __getattr__(name):
    if name == 'VideoPreprocessor':
        from .preprocessing import VideoPreprocessor
        return VideoPreprocessor
    raise AttributeError(name)
