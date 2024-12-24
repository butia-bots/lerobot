import time
from dataclasses import dataclass, field, replace
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
import torch
from lerobot.common.robot_devices.cameras.utils import Camera
import pygame
from tf_transformations import euler_from_matrix
import math

@dataclass
class BorisRobotConfig:
    robot_type: str | None = "boris"
    cameras: dict[str, Camera] = field(default_factory=lambda: {})
    max_relative_target: list[float] | float | None = None
    manipulator_model: str = "wx200"

class BorisRobot:
    def __init__(self, config: BorisRobotConfig | None = None, **kwargs):
        if config is None:
            config = BorisRobotConfig()
        self.config = replace(config, *kwargs)
        self.robot_type = self.config.robot_type
        self.cameras = self.config.cameras
        self.is_connected = False
        self.teleop = None
        self.logs = {}
        self.state_keys = None
        self.action_keys = None
        self.manipulator = None

    def connect(self)->None:
        try:
            self.manipulator = InterbotixManipulatorXS(robot_model=self.config.manipulator_model)
        except:
            print("Could not connect to the arm.")
            raise ConnectionError()
        self.is_connected = True
        for name in self.cameras:
            self.cameras[name].connect()
            self.is_connected = self.is_connected and self.cameras[name].is_connected
        if not self.is_connected:
            print("Could not connect to the cameras, check that all cameras are plugged-in.")
            raise ConnectionError()
        self.run_calibration()

    def run_calibration(self)->None:
        self.manipulator.arm.go_to_home_pose()

    def teleop_step(self, record_data=False):
        if not self.is_connected:
            return ConnectionError()
        if self.teleop is None:
            pygame.joystick.init()
            self.teleop = pygame.joystick.Joystick(id=0)
            self.teleop.init()
        before_read_t = time.perf_counter()
        state = self.get_state()
        delta_pitch = 0.0
        if self.teleop.get_button(pygame.CONTROLLER_BUTTON_DPAD_UP):
            delta_pitch = +0.05
        if self.teleop.get_button(pygame.CONTROLLER_BUTTON_DPAD_DOWN):
            delta_pitch = -0.05
        target_gripper_value = state['gripper_value']
        if self.teleop.get_button(pygame.CONTROLLER_BUTTON_A):
            target_gripper_value -= 0.01
        if self.teleop.get_button(pygame.CONTROLLER_BUTTON_B):
            target_gripper_value += 0.01
        action = dict(
            x=state['x'] + self.teleop.get_axis(pygame.CONTROLLER_AXIS_LEFTY)*0.05,
            y=state['y'] + self.teleop.get_axis(pygame.CONTROLLER_AXIS_LEFTX)*0.05,
            z=state['z'] + self.teleop.get_axis(pygame.CONTROLLER_AXIS_RIGHTY)*0.05,
            roll=state['roll'] + self.teleop.get_axis(pygame.CONTROLLER_AXIS_RIGHTX)*0.05,
            pitch=state['pitch'] + delta_pitch,
            yaw=None,
            gripper_value=target_gripper_value
        )
        action['yaw'] = math.atan2(action['y'], action['x'])
        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t
        before_write_t = time.perf_counter()
        self.manipulator.arm.set_ee_pose_components(
            x=action['x'],
            y=action['y'],
            z=action['z'],
            roll=action['roll'],
            pitch=action['pitch'],
            yaw=None,
            blocking=False
        )
        if target_gripper_value < state['gripper_value']:
            self.manipulator.gripper.grasp(delay=0)
        if target_gripper_value > state["gripper_value"]:
            self.manipulator.gripper.release(delay=0)
        #time.sleep(1/50.0)
        self.logs["write_pos_dt_s"] = time.perf_counter() - before_write_t
        if self.state_keys is None:
            self.state_keys = list(state)
        if not record_data:
            return
        state = torch.as_tensor(state.values())
        action = torch.as_tensor(action.values())
        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            images[name] = torch.from_numpy(images[name])
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t
        obs_dict, action_dict = {}, {}
        obs_dict["observation.state"] = state
        action_dict["action"] = action
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = images[name]
        return obs_dict, action_dict

    def get_state(self):
        arm_ee_pose_matrix = self.manipulator.arm.get_ee_pose()
        x, y, z = arm_ee_pose_matrix[:3,3].flatten()
        roll, pitch, yaw = euler_from_matrix(arm_ee_pose_matrix[:3,:3])
        gripper_value = self.manipulator.gripper.gripper_value
        return dict(
            x=x,
            y=y,
            z=z,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            gripper_value=gripper_value
        )
    
    def capture_observation(self) -> dict:
        before_read_t = time.perf_counter()
        state = self.get_state()
        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t
        if self.state_keys is None:
            self.state_keys = list(state)
        state = torch.as_tensor(list(state.values()))
        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            images[name] = torch.from_numpy(images[name])
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t
        obs_dict = {}
        obs_dict["observation.state"] = state
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = images[name]
        return obs_dict
    
    def send_action(self, action: torch.Tensor) -> torch.Tensor:
        if not self.is_connected:
            raise ConnectionError()
        if self.action_keys is None:
            self.action_keys = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper_value']
        action_dict = dict(zip(self.action_keys, action.tolist(), strict=True))
        state = self.get_state()
        before_write_t = time.perf_counter()
        self.manipulator.arm.set_ee_pose_components(
            x=action_dict['x'],
            y=action_dict['y'],
            z=action_dict['z'],
            roll=action_dict['roll'],
            pitch=action_dict['pitch'],
            yaw=None,
            blocking=False
        )
        if action_dict['gripper_value'] < state['gripper_value']:
            self.manipulator.gripper.grasp(delay=0)
        if action_dict['gripper_value'] > state["gripper_value"]:
            self.manipulator.gripper.release(delay=0)
        #time.sleep(1/50.0)
        self.logs["write_pos_dt_s"] = time.perf_counter() - before_write_t
        return action
    
    def disconnect(self) -> None:
        self.manipulator.shutdown()
        if self.teleop is not None:
            self.teleop.quit()
        if len(self.cameras) > 0:
            for cam in self.cameras.values():
                cam.disconnect()
        self.is_connected = False

    def __del__(self):
        self.disconnect()

    def print_logs(self) -> None:
        pass