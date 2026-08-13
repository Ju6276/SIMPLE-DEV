from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


DEFAULT_SCENE = (
    "/home/d086/fangbaozhong/Sonicstar/SonicStar/wbc/gear_sonic/"
    "data/robot_model/model_data/g1/scene_43dof.xml"
)


def _name_id(model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str) -> int:
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id < 0:
        raise ValueError(f"Missing MuJoCo object {obj_type.name}: {name}")
    return int(obj_id)


def _vec(values: np.ndarray | list[float] | tuple[float, ...]) -> list[float]:
    return [float(x) for x in np.asarray(values, dtype=np.float64).reshape(-1)]


class CupInTrashBinDetector:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        cup_body: str = "cup",
        cup_geom: str = "cup_body",
        bin_body: str = "trash_bin",
        bottom_geom: str = "trash_bin_bottom",
        front_geom: str = "trash_bin_wall_front",
        back_geom: str = "trash_bin_wall_back",
        left_geom: str = "trash_bin_wall_left",
        right_geom: str = "trash_bin_wall_right",
        xy_margin: float = 0.0,
        z_margin: float = 0.02,
        max_speed: float = 0.35,
    ) -> None:
        self.model = model
        self.data = data
        self.cup_body_id = _name_id(model, mujoco.mjtObj.mjOBJ_BODY, cup_body)
        self.bin_body_id = _name_id(model, mujoco.mjtObj.mjOBJ_BODY, bin_body)
        self.cup_geom_id = _name_id(model, mujoco.mjtObj.mjOBJ_GEOM, cup_geom)
        self.bottom_geom_id = _name_id(model, mujoco.mjtObj.mjOBJ_GEOM, bottom_geom)
        self.front_geom_id = _name_id(model, mujoco.mjtObj.mjOBJ_GEOM, front_geom)
        self.back_geom_id = _name_id(model, mujoco.mjtObj.mjOBJ_GEOM, back_geom)
        self.left_geom_id = _name_id(model, mujoco.mjtObj.mjOBJ_GEOM, left_geom)
        self.right_geom_id = _name_id(model, mujoco.mjtObj.mjOBJ_GEOM, right_geom)
        self.xy_margin = float(xy_margin)
        self.z_margin = float(z_margin)
        self.max_speed = float(max_speed)

    def _world_to_bin_local(self, point_world: np.ndarray) -> np.ndarray:
        bin_pos = self.data.xpos[self.bin_body_id]
        bin_xmat = self.data.xmat[self.bin_body_id].reshape(3, 3)
        return bin_xmat.T @ (np.asarray(point_world, dtype=np.float64) - bin_pos)

    def _geom_local_pos(self, geom_id: int) -> np.ndarray:
        body_id = int(self.model.geom_bodyid[geom_id])
        body_pos = self.model.body_pos[body_id]
        body_mat = self.model.body_quat[body_id]
        del body_pos, body_mat
        return np.asarray(self.model.geom_pos[geom_id], dtype=np.float64)

    def _inner_bounds(self) -> dict[str, float]:
        bottom_pos = self._geom_local_pos(self.bottom_geom_id)
        bottom_size = np.asarray(self.model.geom_size[self.bottom_geom_id], dtype=np.float64)
        front_pos = self._geom_local_pos(self.front_geom_id)
        front_size = np.asarray(self.model.geom_size[self.front_geom_id], dtype=np.float64)
        back_pos = self._geom_local_pos(self.back_geom_id)
        back_size = np.asarray(self.model.geom_size[self.back_geom_id], dtype=np.float64)
        left_pos = self._geom_local_pos(self.left_geom_id)
        left_size = np.asarray(self.model.geom_size[self.left_geom_id], dtype=np.float64)
        right_pos = self._geom_local_pos(self.right_geom_id)
        right_size = np.asarray(self.model.geom_size[self.right_geom_id], dtype=np.float64)

        cup_size = np.asarray(self.model.geom_size[self.cup_geom_id], dtype=np.float64)
        cup_radius = float(cup_size[0])

        x_min = float(left_pos[0] + left_size[0] + cup_radius + self.xy_margin)
        x_max = float(right_pos[0] - right_size[0] - cup_radius - self.xy_margin)
        y_min = float(front_pos[1] + front_size[1] + cup_radius + self.xy_margin)
        y_max = float(back_pos[1] - back_size[1] - cup_radius - self.xy_margin)
        z_min = float(bottom_pos[2] + bottom_size[2] - self.z_margin)
        z_max = float(max(front_pos[2] + front_size[2], back_pos[2] + back_size[2]) + self.z_margin)
        return {
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "z_min": z_min,
            "z_max": z_max,
            "cup_radius": cup_radius,
            "cup_half_height": float(cup_size[1]) if cup_size.size > 1 else 0.0,
        }

    def evaluate(self) -> dict[str, Any]:
        mujoco.mj_forward(self.model, self.data)
        cup_world = np.asarray(self.data.xpos[self.cup_body_id], dtype=np.float64)
        bin_world = np.asarray(self.data.xpos[self.bin_body_id], dtype=np.float64)
        cup_local = self._world_to_bin_local(cup_world)
        bounds = self._inner_bounds()

        velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.cup_body_id,
            velocity,
            0,
        )
        linear_speed = float(np.linalg.norm(velocity[3:6]))
        xy_inside = bool(bounds["x_min"] <= cup_local[0] <= bounds["x_max"] and bounds["y_min"] <= cup_local[1] <= bounds["y_max"])
        z_inside = bool(bounds["z_min"] <= cup_local[2] <= bounds["z_max"])
        settled = bool(linear_speed <= self.max_speed)
        success = bool(xy_inside and z_inside and settled)
        return {
            "success": success,
            "xy_inside": xy_inside,
            "z_inside": z_inside,
            "settled": settled,
            "linear_speed": linear_speed,
            "cup_world_pos": _vec(cup_world),
            "trash_bin_world_pos": _vec(bin_world),
            "cup_pos_in_bin_frame": _vec(cup_local),
            "bin_inner_bounds": bounds,
        }


def _set_freejoint_body_pos(model: mujoco.MjModel, data: mujoco.MjData, *, joint_name: str, pos: list[float]) -> None:
    joint_id = _name_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    qpos_addr = int(model.jnt_qposadr[joint_id])
    qvel_addr = int(model.jnt_dofadr[joint_id])
    data.qpos[qpos_addr : qpos_addr + 3] = np.asarray(pos, dtype=np.float64)
    data.qpos[qpos_addr + 3 : qpos_addr + 7] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    data.qvel[qvel_addr : qvel_addr + 6] = 0.0
    mujoco.mj_forward(model, data)


def _print_result(label: str, result: dict[str, Any]) -> None:
    print(json.dumps({"label": label, **result}, ensure_ascii=False, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether the SonicStar cylinder/cup is inside the trash bin.")
    parser.add_argument("--scene", type=str, default=DEFAULT_SCENE)
    parser.add_argument("--cup-joint", type=str, default="cup_freejoint")
    parser.add_argument("--set-cup-pos", type=float, nargs=3, default=None)
    parser.add_argument("--xy-margin", type=float, default=0.0)
    parser.add_argument("--z-margin", type=float, default=0.02)
    parser.add_argument("--max-speed", type=float, default=0.35)
    parser.add_argument("--demo", action="store_true", help="Evaluate initial, inside-bin, and outside-bin positions.")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(Path(args.scene).expanduser().resolve()))
    data = mujoco.MjData(model)
    detector = CupInTrashBinDetector(
        model,
        data,
        xy_margin=float(args.xy_margin),
        z_margin=float(args.z_margin),
        max_speed=float(args.max_speed),
    )
    mujoco.mj_forward(model, data)

    if args.demo:
        _print_result("initial_scene", detector.evaluate())
        bin_pos = np.asarray(data.xpos[detector.bin_body_id], dtype=np.float64)
        _set_freejoint_body_pos(model, data, joint_name=args.cup_joint, pos=_vec(bin_pos + np.asarray([0.0, 0.0, 0.12])))
        _print_result("forced_inside_bin", detector.evaluate())
        _set_freejoint_body_pos(model, data, joint_name=args.cup_joint, pos=_vec(bin_pos + np.asarray([0.35, 0.0, 0.12])))
        _print_result("forced_outside_bin_xy", detector.evaluate())
        _set_freejoint_body_pos(model, data, joint_name=args.cup_joint, pos=_vec(bin_pos + np.asarray([0.0, 0.0, 0.82])))
        _print_result("forced_above_bin", detector.evaluate())
        return

    if args.set_cup_pos is not None:
        _set_freejoint_body_pos(model, data, joint_name=args.cup_joint, pos=[float(x) for x in args.set_cup_pos])
    _print_result("current", detector.evaluate())


if __name__ == "__main__":
    main()
