"""MuJoCo simulation environment and loop for the G1 (and H1) humanoid robots.

DefaultEnv owns the MuJoCo model/data, computes PD torques from Unitree SDK
commands, steps physics, and publishes observations back via the SDK bridge.
BaseSimulator wraps DefaultEnv with rate-limiting and viewer/image update loops.
"""

import json
import os
import pathlib
from pathlib import Path
import pickle
import random
import tempfile
from threading import Lock, Thread
import time
from typing import Dict
import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
import zmq

from gear_sonic.utils.mujoco_sim.metric_utils import check_contact, check_height
from gear_sonic.utils.mujoco_sim.sim_utils import get_subtree_body_names
from gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge import ElasticBand, UnitreeSdk2Bridge
from gear_sonic.utils.mujoco_sim.robot import Robot

GEAR_SONIC_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class DefaultEnv:
    """Base environment class that handles simulation environment setup and step"""

    def __init__(
        self,
        config: Dict[str, any],
        env_name: str = "default",
        camera_configs: Dict[str, any] = {},
        onscreen: bool = False,
        offscreen: bool = False,
        enable_image_publish: bool = False,
    ):
        self.config = config
        self.env_name = env_name
        self.robot = Robot(self.config)
        self.num_body_dof = self.robot.NUM_JOINTS
        self.num_hand_dof = self.robot.NUM_HAND_JOINTS
        self.sim_dt = self.config["SIMULATE_DT"]
        self.obs = None
        self.torques = np.zeros(self.num_body_dof + self.num_hand_dof * 2)
        self.torque_limit = np.array(self.robot.MOTOR_EFFORT_LIMIT_LIST)
        self.camera_configs = camera_configs
        self.camera_randomization_config = self.config.get("CAMERA_RANDOMIZATION", {})
        self._camera_randomization_state = None
        self.robot_start_randomization_config = self.config.get("ROBOT_START_RANDOMIZATION", {})
        self.robot_start_randomization_rng = random.Random()

        if not camera_configs and offscreen and enable_image_publish:
            self.camera_configs = {
                "ego_view": {"height": 480, "width": 640, "mjcf_name": "head_camera"},
            }

        self.reward_lock = Lock()
        self.unitree_bridge = None
        self.onscreen = onscreen

        self.init_scene()
        self.last_reward = 0

        self.offscreen = offscreen
        if self.offscreen:
            self.init_renderers()
        self.image_dt = self.config.get("IMAGE_DT", 0.033333)
        self.image_publish_process = None

    def start_image_publish_subprocess(self, start_method: str = "spawn", camera_port: int = 5555):
        from gear_sonic.utils.mujoco_sim.image_publish_utils import ImagePublishProcess

        if len(self.camera_configs) == 0:
            print(
                "Warning: No camera configs provided, image publishing subprocess will not be started"
            )
            return
        start_method = self.config.get("MP_START_METHOD", "spawn")
        self.image_publish_process = ImagePublishProcess(
            camera_configs=self.camera_configs,
            image_dt=self.image_dt,
            zmq_port=camera_port,
            start_method=start_method,
            verbose=self.config.get("verbose", False),
        )
        self.image_publish_process.start_process()

    def _get_dof_indices_by_class(self):
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".xml") as f:
            mujoco.mj_saveLastXML(f.name, self.mj_model)
            temp_xml_path = f.name

        try:
            tree = ET.parse(temp_xml_path)
            root = tree.getroot()

            joint_class_map = {}
            for joint_element in root.findall(".//joint[@class]"):
                joint_name = joint_element.get("name")
                joint_class = joint_element.get("class")
                if joint_name and joint_class:
                    joint_id = mujoco.mj_name2id(
                        self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
                    )
                    if joint_id != -1:
                        dof_adr = self.mj_model.jnt_dofadr[joint_id]
                        if joint_class not in joint_class_map:
                            joint_class_map[joint_class] = []
                        joint_class_map[joint_class].append(dof_adr)
        finally:
            os.remove(temp_xml_path)

        return joint_class_map

    def _get_default_dof_properties(self):
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".xml") as f:
            mujoco.mj_saveLastXML(f.name, self.mj_model)
            temp_xml_path = f.name

        try:
            tree = ET.parse(temp_xml_path)
            root = tree.getroot()

            default_dof_properties = {}
            for default_element in root.findall(".//default/default[@class]"):
                class_name = default_element.get("class")
                joint_element = default_element.find("joint")
                if class_name and joint_element is not None:
                    properties = {}
                    if "damping" in joint_element.attrib:
                        properties["damping"] = float(joint_element.get("damping"))
                    if "armature" in joint_element.attrib:
                        properties["armature"] = float(joint_element.get("armature"))
                    if "frictionloss" in joint_element.attrib:
                        properties["frictionloss"] = float(joint_element.get("frictionloss"))

                    if properties:
                        default_dof_properties[class_name] = properties
        finally:
            os.remove(temp_xml_path)

        return default_dof_properties

    def init_scene(self):
        """Initialize the default robot scene"""
        xml_path = str(pathlib.Path(GEAR_SONIC_ROOT) / self.config["ROBOT_SCENE"])
        print(f"[MuJoCo] Loading scene XML: {xml_path}", flush=True)
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        body_names = {
            self.mj_model.body(i).name
            for i in range(self.mj_model.nbody)
            if self.mj_model.body(i).name
        }
        print(
            "[MuJoCo] Scene bodies:"
            f" table={'table' in body_names},"
            f" cup={'cup' in body_names}",
            flush=True,
        )
        self.mj_model.opt.timestep = self.sim_dt
        self.torso_index = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self.root_body = "pelvis"
        self.root_body_id = self.mj_model.body(self.root_body).id

        self.joint_class_map = self._get_dof_indices_by_class()

        self.perform_sysid_search = self.config.get("perform_sysid_search", False)
        self._initialize_camera_randomization()
        self.apply_camera_randomization()

        # Check for static root link (fixed base)
        self.use_floating_root_link = "floating_base_joint" in [
            self.mj_model.joint(i).name for i in range(self.mj_model.njnt)
        ]
        self.use_constrained_root_link = "constrained_base_joint" in [
            self.mj_model.joint(i).name for i in range(self.mj_model.njnt)
        ]

        # MuJoCo qpos/qvel arrays start with root DOFs before joint DOFs:
        # floating base has 7 qpos (pos + quat) and 6 qvel (lin + ang velocity)
        if self.use_floating_root_link:
            self.qpos_offset = 7
            self.qvel_offset = 6
        else:
            if self.use_constrained_root_link:
                self.qpos_offset = 1
                self.qvel_offset = 1
            else:
                raise ValueError(
                    "No root link found --"
                    "The absolute static root will make the simulation unstable."
                )

        # Enable the elastic band
        if self.config["ENABLE_ELASTIC_BAND"] and self.use_floating_root_link:
            self.elastic_band = ElasticBand()
            if "g1" in self.config["ROBOT_TYPE"]:
                if self.config["enable_waist"]:
                    self.band_attached_link = self.mj_model.body("pelvis").id
                else:
                    self.band_attached_link = self.mj_model.body("torso_link").id
            elif "h1" in self.config["ROBOT_TYPE"]:
                self.band_attached_link = self.mj_model.body("torso_link").id
            else:
                self.band_attached_link = self.mj_model.body("base_link").id

            if self.onscreen:
                self.viewer = mujoco.viewer.launch_passive(
                    self.mj_model,
                    self.mj_data,
                    key_callback=self.elastic_band.MujuocoKeyCallback,
                    show_left_ui=False,
                    show_right_ui=False,
                )
            else:
                mujoco.mj_forward(self.mj_model, self.mj_data)
                self.viewer = None
        else:
            if self.onscreen:
                self.viewer = mujoco.viewer.launch_passive(
                    self.mj_model, self.mj_data, show_left_ui=False, show_right_ui=False
                )
            else:
                mujoco.mj_forward(self.mj_model, self.mj_data)
                self.viewer = None

        if self.viewer:
            self.viewer.cam.azimuth = 120
            self.viewer.cam.elevation = -30
            self.viewer.cam.distance = 2.0
            self.viewer.cam.lookat = np.array([0, 0, 0.5])
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.viewer.cam.trackbodyid = self.mj_model.body("pelvis").id

        self.body_joint_index = []
        self.left_hand_index = []
        self.right_hand_index = []
        for i in range(self.mj_model.njnt):
            name = self.mj_model.joint(i).name
            if any(
                [
                    part_name in name
                    for part_name in ["hip", "knee", "ankle", "waist", "shoulder", "elbow", "wrist"]
                ]
            ):
                self.body_joint_index.append(i)
            elif "left_hand" in name:
                self.left_hand_index.append(i)
            elif "right_hand" in name:
                self.right_hand_index.append(i)

        assert len(self.body_joint_index) == self.robot.NUM_JOINTS
        assert len(self.left_hand_index) == self.robot.NUM_HAND_JOINTS
        assert len(self.right_hand_index) == self.robot.NUM_HAND_JOINTS

        self.body_joint_index = np.array(self.body_joint_index)
        self.left_hand_index = np.array(self.left_hand_index)
        self.right_hand_index = np.array(self.right_hand_index)

    def init_renderers(self):
        self.renderers = {}
        for camera_name, camera_config in self.camera_configs.items():
            renderer = mujoco.Renderer(
                self.mj_model, height=camera_config["height"], width=camera_config["width"]
            )
            self.renderers[camera_name] = renderer

    def compute_body_torques(self) -> np.ndarray:
        # PD control: tau = tau_ff + kp * (q_des - q) + kd * (dq_des - dq)
        body_torques = np.zeros(self.num_body_dof)
        if self.unitree_bridge is not None and self.unitree_bridge.low_cmd:
            for i in range(self.unitree_bridge.num_body_motor):
                if self.unitree_bridge.use_sensor:
                    body_torques[i] = (
                        self.unitree_bridge.low_cmd.motor_cmd[i].tau
                        + self.unitree_bridge.low_cmd.motor_cmd[i].kp
                        * (self.unitree_bridge.low_cmd.motor_cmd[i].q - self.mj_data.sensordata[i])
                        + self.unitree_bridge.low_cmd.motor_cmd[i].kd
                        * (
                            self.unitree_bridge.low_cmd.motor_cmd[i].dq
                            - self.mj_data.sensordata[i + self.unitree_bridge.num_body_motor]
                        )
                    )
                else:
                    body_torques[i] = (
                        self.unitree_bridge.low_cmd.motor_cmd[i].tau
                        + self.unitree_bridge.low_cmd.motor_cmd[i].kp
                        * (
                            self.unitree_bridge.low_cmd.motor_cmd[i].q
                            - self.mj_data.qpos[self.body_joint_index[i] + self.qpos_offset - 1]
                        )
                        + self.unitree_bridge.low_cmd.motor_cmd[i].kd
                        * (
                            self.unitree_bridge.low_cmd.motor_cmd[i].dq
                            - self.mj_data.qvel[self.body_joint_index[i] + self.qvel_offset - 1]
                        )
                    )
        return body_torques

    def get_head_pose(self) -> np.ndarray:
        root_pos = self.mj_data.body("torso_link").xpos.copy()
        # Reorder quaternion from MuJoCo [w,x,y,z] to scipy [x,y,z,w]
        root_quat = self.mj_data.body("torso_link").xquat.copy()[[1, 2, 3, 0]]
        head_pos = root_pos + Rotation.from_quat(root_quat).apply(np.array([0.0, 0.0, -0.044]))
        return np.concatenate((head_pos, root_quat))

    def get_root_vel(self) -> np.ndarray:
        return self.mj_data.qvel[:6]

    def compute_hand_torques(self) -> np.ndarray:
        left_hand_torques = np.zeros(self.num_hand_dof)
        right_hand_torques = np.zeros(self.num_hand_dof)
        if self.unitree_bridge is not None and self.unitree_bridge.low_cmd:
            for i in range(self.unitree_bridge.num_hand_motor):
                left_hand_torques[i] = (
                    self.unitree_bridge.left_hand_cmd.motor_cmd[i].tau
                    + self.unitree_bridge.left_hand_cmd.motor_cmd[i].kp
                    * (
                        self.unitree_bridge.left_hand_cmd.motor_cmd[i].q
                        - self.mj_data.qpos[self.left_hand_index[i] + self.qpos_offset - 1]
                    )
                    + self.unitree_bridge.left_hand_cmd.motor_cmd[i].kd
                    * (
                        self.unitree_bridge.left_hand_cmd.motor_cmd[i].dq
                        - self.mj_data.qvel[self.left_hand_index[i] + self.qvel_offset - 1]
                    )
                )
                right_hand_torques[i] = (
                    self.unitree_bridge.right_hand_cmd.motor_cmd[i].tau
                    + self.unitree_bridge.right_hand_cmd.motor_cmd[i].kp
                    * (
                        self.unitree_bridge.right_hand_cmd.motor_cmd[i].q
                        - self.mj_data.qpos[self.right_hand_index[i] + self.qpos_offset - 1]
                    )
                    + self.unitree_bridge.right_hand_cmd.motor_cmd[i].kd
                    * (
                        self.unitree_bridge.right_hand_cmd.motor_cmd[i].dq
                        - self.mj_data.qvel[self.right_hand_index[i] + self.qvel_offset - 1]
                    )
                )
        return np.concatenate((left_hand_torques, right_hand_torques))

    def compute_body_qpos(self) -> np.ndarray:
        body_qpos = np.zeros(self.num_body_dof)
        if self.unitree_bridge is not None and self.unitree_bridge.low_cmd:
            for i in range(self.unitree_bridge.num_body_motor):
                body_qpos[i] = self.unitree_bridge.low_cmd.motor_cmd[i].q
        return body_qpos

    def compute_hand_qpos(self) -> np.ndarray:
        hand_qpos = np.zeros(self.num_hand_dof * 2)
        if self.unitree_bridge is not None and self.unitree_bridge.low_cmd:
            for i in range(self.unitree_bridge.num_hand_motor):
                hand_qpos[i] = self.unitree_bridge.left_hand_cmd.motor_cmd[i].q
                hand_qpos[i + self.num_hand_dof] = self.unitree_bridge.right_hand_cmd.motor_cmd[i].q
        return hand_qpos

    def prepare_obs(self) -> Dict[str, any]:
        obs = {}
        if self.use_floating_root_link:
            obs["floating_base_pose"] = self.mj_data.qpos[:7]
            obs["floating_base_vel"] = self.mj_data.qvel[:6]
            obs["floating_base_acc"] = self.mj_data.qacc[:6]
        else:
            obs["floating_base_pose"] = np.zeros(7)
            obs["floating_base_vel"] = np.zeros(6)
            obs["floating_base_acc"] = np.zeros(6)

        obs["secondary_imu_quat"] = self.mj_data.xquat[self.torso_index]

        pose = np.zeros(13)
        torso_link = self.mj_model.body("torso_link").id
        # mj_objectVelocity returns [ang_vel, lin_vel]; swap to [lin_vel, ang_vel]
        mujoco.mj_objectVelocity(
            self.mj_model, self.mj_data, mujoco.mjtObj.mjOBJ_BODY, torso_link, pose[7:13], 1
        )
        pose[7:10], pose[10:13] = (
            pose[10:13],
            pose[7:10].copy(),
        )
        obs["secondary_imu_vel"] = pose[7:13]

        obs["body_q"] = self.mj_data.qpos[self.body_joint_index + 7 - 1]
        obs["body_dq"] = self.mj_data.qvel[self.body_joint_index + 6 - 1]
        obs["body_ddq"] = self.mj_data.qacc[self.body_joint_index + 6 - 1]
        obs["body_tau_est"] = self.mj_data.actuator_force[self.body_joint_index - 1]
        if self.num_hand_dof > 0:
            obs["left_hand_q"] = self.mj_data.qpos[self.left_hand_index + self.qpos_offset - 1]
            obs["left_hand_dq"] = self.mj_data.qvel[self.left_hand_index + self.qvel_offset - 1]
            obs["left_hand_ddq"] = self.mj_data.qacc[self.left_hand_index + self.qvel_offset - 1]
            obs["left_hand_tau_est"] = self.mj_data.actuator_force[self.left_hand_index - 1]
            obs["right_hand_q"] = self.mj_data.qpos[self.right_hand_index + self.qpos_offset - 1]
            obs["right_hand_dq"] = self.mj_data.qvel[self.right_hand_index + self.qvel_offset - 1]
            obs["right_hand_ddq"] = self.mj_data.qacc[self.right_hand_index + self.qvel_offset - 1]
            obs["right_hand_tau_est"] = self.mj_data.actuator_force[self.right_hand_index - 1]
        obs["time"] = self.mj_data.time
        return obs

    def sim_step(self):
        self.obs = self.prepare_obs()
        self.unitree_bridge.PublishLowState(self.obs)
        if self.unitree_bridge.joystick:
            self.unitree_bridge.PublishWirelessController()
        if self.elastic_band:
            if self.elastic_band.enable and self.use_floating_root_link:
                pose = np.concatenate(
                    [
                        self.mj_data.xpos[self.band_attached_link],
                        self.mj_data.xquat[self.band_attached_link],
                        np.zeros(6),
                    ]
                )
                mujoco.mj_objectVelocity(
                    self.mj_model,
                    self.mj_data,
                    mujoco.mjtObj.mjOBJ_BODY,
                    self.band_attached_link,
                    pose[7:13],
                    0,
                )
                pose[7:10], pose[10:13] = pose[10:13], pose[7:10].copy()
                self.mj_data.xfrc_applied[self.band_attached_link] = self.elastic_band.Advance(pose)
            else:
                self.mj_data.xfrc_applied[self.band_attached_link] = np.zeros(6)
        body_torques = self.compute_body_torques()
        hand_torques = self.compute_hand_torques()
        # -1: actuator array is 0-based while joint indices from the model are 1-based
        self.torques[self.body_joint_index - 1] = body_torques
        if self.num_hand_dof > 0:
            self.torques[self.left_hand_index - 1] = hand_torques[: self.num_hand_dof]
            self.torques[self.right_hand_index - 1] = hand_torques[self.num_hand_dof :]

        self.torques = np.clip(self.torques, -self.torque_limit, self.torque_limit)

        if self.config["FREE_BASE"]:
            # Prepend 6 zeros for the floating-base root DOF actuators
            self.mj_data.ctrl = np.concatenate((np.zeros(6), self.torques))
        else:
            self.mj_data.ctrl = self.torques
        mujoco.mj_step(self.mj_model, self.mj_data)

        self.check_fall()

    def apply_perturbation(self, key):
        perturbation_x_body = 0.0
        perturbation_y_body = 0.0
        if key == "up":
            perturbation_x_body = 1.0
        elif key == "down":
            perturbation_x_body = -1.0
        elif key == "left":
            perturbation_y_body = 1.0
        elif key == "right":
            perturbation_y_body = -1.0

        vel_body = np.array([perturbation_x_body, perturbation_y_body, 0.0])
        vel_world = np.zeros(3)
        base_quat = self.mj_data.qpos[3:7]
        mujoco.mju_rotVecQuat(vel_world, vel_body, base_quat)

        self.mj_data.qvel[0] += vel_world[0]
        self.mj_data.qvel[1] += vel_world[1]
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def update_viewer(self):
        if self.viewer is not None:
            self.viewer.sync()

    def update_viewer_camera(self):
        if self.viewer is not None:
            if self.viewer.cam.type == mujoco.mjtCamera.mjCAMERA_TRACKING:
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            else:
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING

    def update_reward(self):
        with self.reward_lock:
            self.last_reward = 0

    def get_reward(self):
        with self.reward_lock:
            return self.last_reward

    def set_unitree_bridge(self, unitree_bridge):
        self.unitree_bridge = unitree_bridge

    def get_privileged_obs(self):
        return {}

    def update_render_caches(self):
        render_caches = {}
        for camera_name, camera_config in self.camera_configs.items():
            renderer = self.renderers[camera_name]
            if "params" in camera_config:
                renderer.update_scene(self.mj_data, camera=camera_config["params"])
            elif "mjcf_name" in camera_config:
                renderer.update_scene(self.mj_data, camera=camera_config["mjcf_name"])
            else:
                renderer.update_scene(self.mj_data, camera=camera_name)
            render_caches[camera_name + "_image"] = renderer.render()

        if self.image_publish_process is not None:
            self.image_publish_process.update_shared_memory(render_caches)

        return render_caches

    def handle_keyboard_button(self, key):
        if self.elastic_band:
            self.elastic_band.handle_keyboard_button(key)

        if key == "backspace":
            self.reset()
        if key == "v":
            self.update_viewer_camera()
        if key in ["up", "down", "left", "right"]:
            self.apply_perturbation(key)

    def check_fall(self):
        self.fall = False
        if self.mj_data.qpos[2] < 0.2:
            self.fall = True
            print(f"Warning: Robot has fallen, height: {self.mj_data.qpos[2]:.3f} m")

        if self.fall:
            self.reset()

    def check_self_collision(self):
        robot_bodies = get_subtree_body_names(self.mj_model, self.mj_model.body(self.root_body).id)
        self_collision, contact_bodies = check_contact(
            self.mj_model, self.mj_data, robot_bodies, robot_bodies, return_all_contact_bodies=True
        )
        if self_collision:
            print(f"Warning: Self-collision detected: {contact_bodies}")
        return self_collision

    def _initialize_camera_randomization(self):
        cfg = self.camera_randomization_config
        if not cfg.get("enabled", False):
            return

        camera_name = cfg.get("camera_name", "head_camera")
        camera_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if camera_id == -1:
            print(f"[MuJoCo] Camera randomization skipped: camera '{camera_name}' not found")
            return

        base_pos = np.array(self.mj_model.cam_pos[camera_id], dtype=np.float64).copy()
        base_quat = np.array(self.mj_model.cam_quat[camera_id], dtype=np.float64).copy()
        self._camera_randomization_state = {
            "camera_name": camera_name,
            "camera_id": camera_id,
            "base_pos": base_pos,
            "base_quat": base_quat,
            "rng": random.Random(),
        }
        print(
            "[MuJoCo] Camera randomization enabled:"
            f" camera={camera_name}"
            f" pos_jitter=({cfg.get('pos_jitter_x', 0.0):.4f},"
            f" {cfg.get('pos_jitter_y', 0.0):.4f},"
            f" {cfg.get('pos_jitter_z', 0.0):.4f})"
            f" rpy_jitter=({cfg.get('roll_jitter', 0.0):.4f},"
            f" {cfg.get('pitch_jitter', 0.0):.4f},"
            f" {cfg.get('yaw_jitter', 0.0):.4f})",
            flush=True,
        )

    def apply_camera_randomization(self):
        state = self._camera_randomization_state
        if state is None:
            return

        cfg = self.camera_randomization_config
        rng = state["rng"]

        pos_delta = np.array(
            [
                rng.uniform(-cfg.get("pos_jitter_x", 0.0), cfg.get("pos_jitter_x", 0.0)),
                rng.uniform(-cfg.get("pos_jitter_y", 0.0), cfg.get("pos_jitter_y", 0.0)),
                rng.uniform(-cfg.get("pos_jitter_z", 0.0), cfg.get("pos_jitter_z", 0.0)),
            ],
            dtype=np.float64,
        )
        euler_delta = np.array(
            [
                rng.uniform(-cfg.get("roll_jitter", 0.0), cfg.get("roll_jitter", 0.0)),
                rng.uniform(-cfg.get("pitch_jitter", 0.0), cfg.get("pitch_jitter", 0.0)),
                rng.uniform(-cfg.get("yaw_jitter", 0.0), cfg.get("yaw_jitter", 0.0)),
            ],
            dtype=np.float64,
        )

        delta_quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_euler2Quat(delta_quat, euler_delta, "xyz")

        new_quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mulQuat(new_quat, state["base_quat"], delta_quat)

        camera_id = state["camera_id"]
        self.mj_model.cam_pos[camera_id] = state["base_pos"] + pos_delta
        self.mj_model.cam_quat[camera_id] = new_quat

        print(
            "[MuJoCo] Camera randomized:"
            f" camera={state['camera_name']}"
            f" pos_delta=({pos_delta[0]:+.4f}, {pos_delta[1]:+.4f}, {pos_delta[2]:+.4f})"
            f" euler_delta=({euler_delta[0]:+.4f}, {euler_delta[1]:+.4f}, {euler_delta[2]:+.4f})",
            flush=True,
        )

    def apply_robot_start_randomization(self):
        cfg = self.robot_start_randomization_config
        if not cfg.get("enabled", False) or not self.use_floating_root_link:
            return

        min_meters = float(cfg.get("min_meters", 0.02))
        max_meters = float(cfg.get("max_meters", 0.07))
        if min_meters < 0.0 or max_meters < min_meters:
            print(
                "[MuJoCo] Robot start randomization skipped: "
                f"invalid range min={min_meters:.4f}, max={max_meters:.4f}",
                flush=True,
            )
            return

        distance = self.robot_start_randomization_rng.uniform(min_meters, max_meters)
        direction = self.robot_start_randomization_rng.choice((-1.0, 1.0))
        offset_body = np.array([direction * distance, 0.0, 0.0], dtype=np.float64)
        offset_world = np.zeros(3, dtype=np.float64)
        mujoco.mju_rotVecQuat(offset_world, offset_body, self.mj_data.qpos[3:7])
        self.mj_data.qpos[:3] += offset_world

        print(
            "[MuJoCo] Robot start randomized: "
            f"body_x_offset={offset_body[0]:+.4f}m "
            f"world_offset=({offset_world[0]:+.4f}, {offset_world[1]:+.4f}, {offset_world[2]:+.4f})",
            flush=True,
        )

    def reset(self):
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        self.apply_robot_start_randomization()
        self.apply_camera_randomization()
        mujoco.mj_forward(self.mj_model, self.mj_data)


class BaseSimulator:
    """Base simulator class that handles initialization and running of simulations"""

    def __init__(
        self, config: Dict[str, any], env_name: str = "default", redis_client=None, **kwargs
    ):
        self.config = config
        self.env_name = env_name
        self.redis_client = redis_client
        if self.redis_client is not None:
            self.redis_client.set("push_left_hand", "false")
            self.redis_client.set("push_right_hand", "false")
            self.redis_client.set("push_torso", "false")

        # Create rate objects
        self.sim_dt = self.config["SIMULATE_DT"]
        self.reward_dt = self.config.get("REWARD_DT", 0.02)
        self.image_dt = self.config.get("IMAGE_DT", 0.033333)
        self.viewer_dt = self.config.get("VIEWER_DT", 0.02)
        self._running = True
        self._sim_lock = Lock()
        self._eval_server_thread = None
        self._eval_server_context = None
        self.sim_step_counter = 0
        self.reset_count = 0

        self.robot = Robot(self.config)

        # Create the environment
        if env_name == "default":
            self.sim_env = DefaultEnv(config, env_name, **kwargs)
        else:
            raise ValueError(
                f"Invalid environment name: {env_name}. "
                f"Only 'default' is supported in this minimal build."
            )

        try:
            # run_sim_loop may initialize ChannelFactory before constructing BaseSimulator.
            # Skip duplicate init to avoid CycloneDDS domain conflicts.
            if not self.config.get("__channel_factory_initialized", False):
                if self.config.get("INTERFACE", None):
                    ChannelFactoryInitialize(self.config["DOMAIN_ID"], self.config["INTERFACE"])
                else:
                    ChannelFactoryInitialize(self.config["DOMAIN_ID"])
                self.config["__channel_factory_initialized"] = True
        except Exception as e:
            print(f"Note: Channel factory initialization attempt: {e}")

        self.init_unitree_bridge()
        self.sim_env.set_unitree_bridge(self.unitree_bridge)

        self.init_subscriber()
        self.init_publisher()

        self.sim_thread = None
        if self.config.get("EVAL_SERVER_ENABLED", False):
            self.start_eval_server()

    def start_as_thread(self):
        self.sim_thread = Thread(target=self.start)
        self.sim_thread.start()

    def start_image_publish_subprocess(self, start_method: str = "spawn", camera_port: int = 5555):
        self.sim_env.start_image_publish_subprocess(start_method, camera_port)

    def init_subscriber(self):
        pass

    def init_publisher(self):
        pass

    def init_unitree_bridge(self):
        self.unitree_bridge = UnitreeSdk2Bridge(self.config)
        if self.config["USE_JOYSTICK"]:
            self.unitree_bridge.SetupJoystick(
                device_id=self.config["JOYSTICK_DEVICE"], js_type=self.config["JOYSTICK_TYPE"]
            )

    def start(self):
        """Main simulation loop"""
        sim_cnt = 0
        ts = time.time()

        try:
            while self._running and (
                (self.sim_env.viewer and self.sim_env.viewer.is_running())
                or (self.sim_env.viewer is None)
            ):
                step_start = time.monotonic()

                with self._sim_lock:
                    self.sim_env.sim_step()
                    self.sim_step_counter = sim_cnt
                    now = time.time()
                    if now - ts > 1 / 10.0 and self.redis_client is not None:
                        head_pose = self.sim_env.get_head_pose()
                        self.redis_client.set("head_pos", pickle.dumps(head_pose[:3]))
                        self.redis_client.set("head_quat", pickle.dumps(head_pose[3:]))
                        ts = now

                    if sim_cnt % int(self.viewer_dt / self.sim_dt) == 0:
                        self.sim_env.update_viewer()

                    if sim_cnt % int(self.reward_dt / self.sim_dt) == 0:
                        self.sim_env.update_reward()

                    if sim_cnt % int(self.image_dt / self.sim_dt) == 0:
                        self.sim_env.update_render_caches()

                # Simple rate limiter (replaces ROS rate)
                elapsed = time.monotonic() - step_start
                sleep_time = self.sim_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                sim_cnt += 1
        except KeyboardInterrupt:
            print("Simulator interrupted by user.")
        finally:
            self.close()

    def __del__(self):
        self.close()

    def reset(self):
        with self._sim_lock:
            self.sim_env.reset()
            self.reset_count += 1

    def _name_id(self, obj_type: mujoco.mjtObj, name: str) -> int:
        obj_id = mujoco.mj_name2id(self.sim_env.mj_model, obj_type, name)
        if obj_id < 0:
            raise ValueError(f"Missing MuJoCo object {obj_type.name}: {name}")
        return int(obj_id)

    @staticmethod
    def _vec(values) -> list[float]:
        return [float(x) for x in np.asarray(values, dtype=np.float64).reshape(-1)]

    def _world_to_body_local(self, point_world: np.ndarray, body_id: int) -> np.ndarray:
        body_pos = self.sim_env.mj_data.xpos[body_id]
        body_xmat = self.sim_env.mj_data.xmat[body_id].reshape(3, 3)
        return body_xmat.T @ (np.asarray(point_world, dtype=np.float64) - body_pos)

    def _geom_local_pos(self, geom_id: int) -> np.ndarray:
        return np.asarray(self.sim_env.mj_model.geom_pos[geom_id], dtype=np.float64)

    def _cup_bin_inner_bounds(self, ids: dict[str, int], xy_margin: float, z_margin: float) -> dict[str, float]:
        model = self.sim_env.mj_model
        bottom_pos = self._geom_local_pos(ids["bottom_geom"])
        bottom_size = np.asarray(model.geom_size[ids["bottom_geom"]], dtype=np.float64)
        front_pos = self._geom_local_pos(ids["front_geom"])
        front_size = np.asarray(model.geom_size[ids["front_geom"]], dtype=np.float64)
        back_pos = self._geom_local_pos(ids["back_geom"])
        back_size = np.asarray(model.geom_size[ids["back_geom"]], dtype=np.float64)
        left_pos = self._geom_local_pos(ids["left_geom"])
        left_size = np.asarray(model.geom_size[ids["left_geom"]], dtype=np.float64)
        right_pos = self._geom_local_pos(ids["right_geom"])
        right_size = np.asarray(model.geom_size[ids["right_geom"]], dtype=np.float64)
        cup_size = np.asarray(model.geom_size[ids["cup_geom"]], dtype=np.float64)
        cup_radius = float(cup_size[0])

        return {
            "x_min": float(left_pos[0] + left_size[0] + cup_radius + xy_margin),
            "x_max": float(right_pos[0] - right_size[0] - cup_radius - xy_margin),
            "y_min": float(front_pos[1] + front_size[1] + cup_radius + xy_margin),
            "y_max": float(back_pos[1] - back_size[1] - cup_radius - xy_margin),
            "z_min": float(bottom_pos[2] + bottom_size[2] - z_margin),
            "z_max": float(max(front_pos[2] + front_size[2], back_pos[2] + back_size[2]) + z_margin),
            "cup_radius": cup_radius,
            "cup_half_height": float(cup_size[1]) if cup_size.size > 1 else 0.0,
        }

    def eval_status(
        self,
        *,
        xy_margin: float = 0.0,
        z_margin: float = 0.02,
        max_speed: float = 0.35,
    ) -> dict:
        with self._sim_lock:
            model = self.sim_env.mj_model
            data = self.sim_env.mj_data
            ids = {
                "cup_body": self._name_id(mujoco.mjtObj.mjOBJ_BODY, "cup"),
                "bin_body": self._name_id(mujoco.mjtObj.mjOBJ_BODY, "trash_bin"),
                "cup_geom": self._name_id(mujoco.mjtObj.mjOBJ_GEOM, "cup_body"),
                "bottom_geom": self._name_id(mujoco.mjtObj.mjOBJ_GEOM, "trash_bin_bottom"),
                "front_geom": self._name_id(mujoco.mjtObj.mjOBJ_GEOM, "trash_bin_wall_front"),
                "back_geom": self._name_id(mujoco.mjtObj.mjOBJ_GEOM, "trash_bin_wall_back"),
                "left_geom": self._name_id(mujoco.mjtObj.mjOBJ_GEOM, "trash_bin_wall_left"),
                "right_geom": self._name_id(mujoco.mjtObj.mjOBJ_GEOM, "trash_bin_wall_right"),
            }
            mujoco.mj_forward(model, data)
            cup_world = np.asarray(data.xpos[ids["cup_body"]], dtype=np.float64)
            bin_world = np.asarray(data.xpos[ids["bin_body"]], dtype=np.float64)
            cup_local = self._world_to_body_local(cup_world, ids["bin_body"])
            bounds = self._cup_bin_inner_bounds(ids, float(xy_margin), float(z_margin))

            velocity = np.zeros(6, dtype=np.float64)
            mujoco.mj_objectVelocity(
                model,
                data,
                mujoco.mjtObj.mjOBJ_BODY,
                ids["cup_body"],
                velocity,
                0,
            )
            linear_speed = float(np.linalg.norm(velocity[3:6]))
            xy_inside = bool(
                bounds["x_min"] <= cup_local[0] <= bounds["x_max"]
                and bounds["y_min"] <= cup_local[1] <= bounds["y_max"]
            )
            z_inside = bool(bounds["z_min"] <= cup_local[2] <= bounds["z_max"])
            settled = bool(linear_speed <= float(max_speed))
            success = bool(xy_inside and z_inside and settled)
            return {
                "success": success,
                "xy_inside": xy_inside,
                "z_inside": z_inside,
                "settled": settled,
                "linear_speed": linear_speed,
                "cup_world_pos": self._vec(cup_world),
                "trash_bin_world_pos": self._vec(bin_world),
                "cup_pos_in_bin_frame": self._vec(cup_local),
                "bin_inner_bounds": bounds,
                "robot_qpos_head": self._vec(data.qpos[: min(16, data.qpos.size)]),
                "robot_qvel_head": self._vec(data.qvel[: min(16, data.qvel.size)]),
                "sim_time": float(data.time),
                "sim_step": int(self.sim_step_counter),
                "reset_count": int(self.reset_count),
            }

    def start_eval_server(self):
        host = str(self.config.get("EVAL_SERVER_HOST", "127.0.0.1"))
        port = int(self.config.get("EVAL_SERVER_PORT", 5561))
        self._eval_server_context = zmq.Context()

        def loop():
            socket = self._eval_server_context.socket(zmq.REP)
            socket.setsockopt(zmq.LINGER, 0)
            endpoint = f"tcp://{host}:{port}"
            socket.bind(endpoint)
            print(f"[MuJoCo Eval] Server listening on {endpoint}", flush=True)
            poller = zmq.Poller()
            poller.register(socket, zmq.POLLIN)
            while self._running:
                try:
                    events = dict(poller.poll(100))
                except zmq.ContextTerminated:
                    break
                if socket not in events:
                    continue
                try:
                    request = socket.recv_json()
                    command = str(request.get("command", "status"))
                    kwargs = {
                        "xy_margin": float(request.get("xy_margin", 0.0)),
                        "z_margin": float(request.get("z_margin", 0.02)),
                        "max_speed": float(request.get("max_speed", 0.35)),
                    }
                    if command == "reset":
                        self.reset()
                        payload = {"ok": True, "command": command, **self.eval_status(**kwargs)}
                    elif command == "status":
                        payload = {"ok": True, "command": command, **self.eval_status(**kwargs)}
                    elif command == "key":
                        key = str(request.get("key", ""))
                        if not key:
                            payload = {"ok": False, "error": "missing key"}
                        else:
                            self.handle_keyboard_button(key)
                            payload = {"ok": True, "command": command, "key": key}
                    elif command == "close":
                        self._running = False
                        payload = {"ok": True, "command": command}
                    else:
                        payload = {"ok": False, "error": f"unknown command: {command}"}
                except Exception as exc:
                    payload = {"ok": False, "error": repr(exc)}
                socket.send_string(json.dumps(payload, sort_keys=True))
            try:
                socket.close()
            except Exception:
                pass

        self._eval_server_thread = Thread(target=loop, daemon=True)
        self._eval_server_thread.start()

    def close(self):
        self._running = False
        try:
            if self.sim_env.image_publish_process is not None:
                self.sim_env.image_publish_process.stop()
            if self.sim_env.viewer is not None:
                self.sim_env.viewer.close()
            if self._eval_server_context is not None:
                self._eval_server_context.destroy(linger=0)
                self._eval_server_context = None
        except Exception as e:
            print(f"Warning during close: {e}")

    def get_privileged_obs(self):
        return self.sim_env.get_privileged_obs()

    def handle_keyboard_button(self, key):
        with self._sim_lock:
            self.sim_env.handle_keyboard_button(key)
