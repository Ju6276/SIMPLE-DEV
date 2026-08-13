"""Create a copy of scene_43dof.xml with a randomized cup position."""

from __future__ import annotations

import argparse
import random
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-scene", required=True, help="Source MuJoCo scene XML path")
    parser.add_argument("--output-scene", required=True, help="Output MuJoCo scene XML path")
    parser.add_argument("--x-min", type=float, required=True, help="Random cup x lower bound")
    parser.add_argument("--x-max", type=float, required=True, help="Random cup x upper bound")
    parser.add_argument("--y-min", type=float, required=True, help="Random cup y lower bound")
    parser.add_argument("--y-max", type=float, required=True, help="Random cup y upper bound")
    parser.add_argument(
        "--z",
        type=float,
        default=None,
        help="Optional cup center z value. Defaults to the original scene z.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible cup placement.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    input_scene = Path(args.input_scene)
    output_scene = Path(args.output_scene)
    output_scene.parent.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(input_scene)
    root = tree.getroot()

    # When the randomized scene is written to /tmp, relative include paths
    # from the original scene would otherwise break. Rebase them to absolute
    # paths so MuJoCo can still resolve nested robot XMLs correctly.
    for include_node in root.findall("include"):
        include_file = include_node.get("file")
        if include_file and not Path(include_file).is_absolute():
            include_node.set("file", str((input_scene.parent / include_file).resolve()))

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"Missing worldbody in scene XML: {input_scene}")

    cup_body = worldbody.find("body[@name='cup']")
    if cup_body is None:
        raise ValueError(f"Missing cup body in scene XML: {input_scene}")

    current_pos = [float(v) for v in cup_body.attrib["pos"].split()]
    cup_x = rng.uniform(args.x_min, args.x_max)
    cup_y = rng.uniform(args.y_min, args.y_max)
    cup_z = current_pos[2] if args.z is None else args.z
    cup_body.set("pos", f"{cup_x:.6f} {cup_y:.6f} {cup_z:.6f}")

    tree.write(output_scene, encoding="utf-8", xml_declaration=False)

    print(
        "[randomize_scene_43dof_cup] "
        f"output={output_scene} cup_pos=({cup_x:.6f}, {cup_y:.6f}, {cup_z:.6f})"
    )


if __name__ == "__main__":
    main()
