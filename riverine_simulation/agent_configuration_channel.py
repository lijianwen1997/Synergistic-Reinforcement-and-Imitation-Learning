from mlagents_envs.side_channel import SideChannel, OutgoingMessage, IncomingMessage
from mlagents_envs.exception import (
    UnityCommunicationException,
    UnitySideChannelException,
)
import uuid
from typing import NamedTuple, Optional


class AgentConfig(NamedTuple):
    vertical_delta: Optional[float]  # meter
    long_delta: Optional[float]  # meter
    lat_delta: Optional[float]  # meter
    rotation_delta: Optional[float]  # degree
    min_above_height: Optional[float]  # meter
    max_above_height: Optional[float]  # meter
    max_angle_offset: Optional[float]  # degree
    pitch_angle: Optional[float]  # degree
    max_idle_steps: Optional[int]

    @staticmethod
    def default_config():
        return AgentConfig(1, 1, 0.5, 10, 1, 15, 15, 20, 50)


class AgentConfigurationChannel(SideChannel):

    def __init__(self) -> None:
        super().__init__(uuid.UUID("e951342c-4f7e-11ea-b238-784f43874321"))

    def on_message_received(self, msg: IncomingMessage) -> None:
        """
        Is called by the environment to the side channel. Can be called
        multiple times per step if multiple messages are meant for that
        SideChannel.
        Note that Python should never receive an agent configuration from
        Unity
        """
        raise UnityCommunicationException(
            "The AgentConfigurationChannel received a message from Unity, "
            + "this should not have happened."
        )

    def set_configuration_parameters(
            self,
            vertical_delta: Optional[float] = None,
            long_delta: Optional[float] = None,
            lat_delta: Optional[float] = None,
            rotation_delta: Optional[float] = None,
            min_above_height: Optional[float] = None,
            max_above_height: Optional[float] = None,
            max_angle_offset: Optional[float] = None,
            pitch_angle: Optional[float] = None,
            max_idle_steps: Optional[int] = None):
        '''
        Sets the agent configuration. Takes as input the configurations of the agent.
        :param vertical_delta:
        :param long_delta:
        :param lat_delta:
        :param rotation_delta:
        :param min_above_height:
        :param max_above_height:
        :param max_angle_offset:
        :param pitch_angle:
        :param max_idle_steps:
        :return:
        '''
        if vertical_delta is not None:
            msg = OutgoingMessage()
            msg.write_int32(0)
            msg.write_float32(vertical_delta)
            super().queue_message_to_send(msg)
        if long_delta is not None:
            msg = OutgoingMessage()
            msg.write_int32(1)
            msg.write_float32(long_delta)
            super().queue_message_to_send(msg)
        if lat_delta is not None:
            msg = OutgoingMessage()
            msg.write_int32(2)
            msg.write_float32(lat_delta)
            super().queue_message_to_send(msg)
        if rotation_delta is not None:
            msg = OutgoingMessage()
            msg.write_int32(3)
            msg.write_float32(rotation_delta)
            super().queue_message_to_send(msg)
        if min_above_height is not None:
            msg = OutgoingMessage()
            msg.write_int32(4)
            msg.write_float32(min_above_height)
            super().queue_message_to_send(msg)
        if max_above_height is not None:
            msg = OutgoingMessage()
            msg.write_int32(5)
            msg.write_float32(max_above_height)
            super().queue_message_to_send(msg)
        if max_angle_offset is not None:
            msg = OutgoingMessage()
            msg.write_int32(6)
            msg.write_float32(max_angle_offset)
            super().queue_message_to_send(msg)
        if pitch_angle is not None:
            msg = OutgoingMessage()
            msg.write_int32(7)
            msg.write_float32(pitch_angle)
            super().queue_message_to_send(msg)
        if max_idle_steps is not None:
            msg = OutgoingMessage()
            msg.write_int32(8)
            msg.write_int32(max_idle_steps)
            super().queue_message_to_send(msg)

    def set_configuration(self, config: AgentConfig) -> None:
        """
        Sets the agent configuration. Takes as input an AgentConfig.
        """
        self.set_configuration_parameters(**config._asdict())



