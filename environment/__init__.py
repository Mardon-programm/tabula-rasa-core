"""
TR-Core Environments.

Attribute task environments for behavioral program induction experiments.
"""

from environment.attribute_tasks import *
from environment.attribute_tasks_v3 import *
from environment.attribute_tasks_v4 import *
from environment.sandbox_sim import *
from environment.gymnasium_env import (
    AttributeTaskEnv,
    AttributeTaskConfig,
    make_attribute_task_v3,
    make_attribute_task_v4,
    register_envs,
)

__all__ = [
    "AttributeTaskEnv",
    "AttributeTaskConfig",
    "make_attribute_task_v3",
    "make_attribute_task_v4",
    "register_envs",
]