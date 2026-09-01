from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class SensorReading:
    sensor: str
    value: Any
    timestamp: float


@dataclass(slots=True)
class MotorCommand:
    actuator: str
    value: float


class Embodiment(ABC):
    @abstractmethod
    def read_sensors(self) -> list[SensorReading]:
        raise NotImplementedError

    @abstractmethod
    def execute(
        self,
        command: MotorCommand,
    ) -> bool:
        raise NotImplementedError


class VirtualEmbodiment(Embodiment):
    """
    Simulator-backed embodiment.

    Used before physical hardware integration.
    """

    def __init__(self) -> None:
        self.commands: list[MotorCommand] = []
        self.sensors: list[SensorReading] = []

    def read_sensors(self) -> list[SensorReading]:
        return list(self.sensors)

    def execute(
        self,
        command: MotorCommand,
    ) -> bool:
        self.commands.append(command)
        return True

    def add_sensor_reading(
        self,
        reading: SensorReading,
    ) -> None:
        self.sensors.append(reading)

    def snapshot(self) -> dict[str, Any]:
        return {
            "commands_executed": len(self.commands),
            "sensor_readings": len(self.sensors),
        }