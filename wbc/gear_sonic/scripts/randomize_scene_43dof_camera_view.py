"""Create a copy of scene_43dof.xml with a lightly randomized MuJoCo camera view."""

from __future__ import annotations

import argparse
import random
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-scene", required=True, help="Source MuJoCo scene XML path")
    parser.add_argument("--output-scene", required=True, help="Output MuJoCo scene XML path")
    parser.add_argument(
        "--output-robot-xml",
        required=True,
        help="Output MuJoCo robot XML path referenced by the randomized scene",
    )
    parser.add_argument(
        "--camera-name",
        default="head_camera",
        help="MuJoCo camera name to perturb inside the robot XML",
    )
    parser.add_argument(
        "--pos-jitter",
        type=float,
        default=0.015,
        help="Uniform +/- position jitter in meters for x/y/z when per-axis jitter is unset.",
    )
    parser.add_argument(
        "--pos-jitter-x",
        type=float,
        default=None,
        help="Optional uniform +/- position jitter in meters for x.",
    )
    parser.add_argument(
        "--pos-jitter-y",
        type=float,
        default=None,
        help="Optional uniform +/- position jitter in meters for y.",
    )
    parser.add_argument(
        "--pos-jitter-z",
        type=float,
        default=None,
        help="Optional uniform +/- position jitter in meters for z.",
    )
    parser.add_argument(
        "--roll-jitter",
        type=float,
        default=0.05,
        help="Uniform +/- roll jitter in radians.",
    )
    parser.add_argument(
        "--pitch-jitter",
        type=float,
        default=0.08,
        help="Uniform +/- pitch jitter in radians.",
    )
    parser.add_argument(
        "--yaw-jitter",
        type=float,
        default=0.08,
        help="Uniform +/- yaw jitter in radians.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible camera perturbation.",
    )
    return parser.parse_args()


def format_vec(values: list[float]) -> str:
    return " ".join(f"{value:.6f}" for value in values)


def resolve_include_path(input_scene: Path, include_file: str) -> Path:
    include_path = Path(include_file)
    if include_path.is_absolute():
        return include_path
    return (input_scene.parent / include_path).resolve()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    input_scene = Path(args.input_scene)
    output_scene = Path(args.output_scene)
    output_robot_xml = Path(args.output_robot_xml)
    output_scene.parent.mkdir(parents=True, exist_ok=True)
    output_robot_xml.parent.mkdir(parents=True, exist_ok=True)

    scene_tree = ET.parse(input_scene)
    scene_root = scene_tree.getroot()

    include_node = scene_root.find("include")
    if include_node is None:
        raise ValueError(f"Missing include in scene XML: {input_scene}")

    include_file = include_node.get("file")
    if not include_file:
        raise ValueError(f"Scene include is missing file attribute: {input_scene}")

    input_robot_xml = resolve_include_path(input_scene, include_file)

    robot_tree = ET.parse(input_robot_xml)
    robot_root = robot_tree.getroot()

    camera_node = robot_root.find(f".//camera[@name='{args.camera_name}']")
    if camera_node is None:
        raise ValueError(f"Missing camera '{args.camera_name}' in robot XML: {input_robot_xml}")

    base_pos = [float(value) for value in camera_node.get("pos", "0 0 0").split()]
    base_euler = [float(value) for value in camera_node.get("euler", "0 0 0").split()]

    pos_jitter_x = args.pos_jitter if args.pos_jitter_x is None else args.pos_jitter_x
    pos_jitter_y = args.pos_jitter if args.pos_jitter_y is None else args.pos_jitter_y
    pos_jitter_z = args.pos_jitter if args.pos_jitter_z is None else args.pos_jitter_z

    pos_delta = [
        rng.uniform(-pos_jitter_x, pos_jitter_x),
        rng.uniform(-pos_jitter_y, pos_jitter_y),
        rng.uniform(-pos_jitter_z, pos_jitter_z),
    ]
    euler_delta = [
        rng.uniform(-args.roll_jitter, args.roll_jitter),
        rng.uniform(-args.pitch_jitter, args.pitch_jitter),
        rng.uniform(-args.yaw_jitter, args.yaw_jitter),
    ]

    new_pos = [base + delta for base, delta in zip(base_pos, pos_delta)]
    new_euler = [base + delta for base, delta in zip(base_euler, euler_delta)]

    camera_node.set("pos", format_vec(new_pos))
    camera_node.set("euler", format_vec(new_euler))

    robot_tree.write(output_robot_xml, encoding="utf-8", xml_declaration=False)

    include_node.set("file", str(output_robot_xml.resolve()))
    scene_tree.write(output_scene, encoding="utf-8", xml_declaration=False)

    print(
        "[randomize_scene_43dof_camera_view] "
        f"output_scene={output_scene} "
        f"output_robot_xml={output_robot_xml} "
        f"camera={args.camera_name} "
        f"pos={camera_node.get('pos')} "
        f"euler={camera_node.get('euler')}"
    )


if __name__ == "__main__":
    main()
