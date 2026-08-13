"""Add an overhead vertical light above the table in scene_43dof.xml."""

from __future__ import annotations

import argparse
import random
import xml.etree.ElementTree as ET
from pathlib import Path

# Table body is at (0.85, 0.00, 0.38); table_top center is +0.36 in z.
TABLE_CENTER_X = 0.85
TABLE_CENTER_Y = 0.00
TABLE_TOP_HALF_X = 0.30
TABLE_TOP_HALF_Y = 0.45
TABLE_TOP_Z = 0.74
OVERHEAD_LIGHT_Z = 2.00
VERTICAL_DIR = (0.0, 0.0, -1.0)

MANAGED_LIGHT_NAMES = {
    "table_overhead_light",
    "studio_key_light",
    "studio_fill_light",
    "studio_rim_light",
    "studio_table_spot",
    "studio_ceiling_light",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-scene", required=True, help="Source MuJoCo scene XML path")
    parser.add_argument("--output-scene", required=True, help="Output MuJoCo scene XML path")
    parser.add_argument(
        "--randomize",
        action="store_true",
        help="Randomize overhead light xy position on the table and light intensity.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible lighting.",
    )
    return parser.parse_args()


def rebase_include_paths(root: ET.Element, input_scene: Path) -> None:
    for include_node in root.findall("include"):
        include_file = include_node.get("file")
        if include_file and not Path(include_file).is_absolute():
            include_node.set("file", str((input_scene.parent / include_file).resolve()))


def format_vec3(values: tuple[float, float, float]) -> str:
    return f"{values[0]:.6f} {values[1]:.6f} {values[2]:.6f}"


def scale_rgb(
    rgb: tuple[float, float, float],
    rng: random.Random,
    intensity_scale_range: tuple[float, float],
    tint_range: float,
) -> tuple[float, float, float]:
    intensity = rng.uniform(*intensity_scale_range)
    tint = [rng.uniform(-tint_range, tint_range) for _ in range(3)]
    return tuple(
        min(1.0, max(0.05, channel * intensity + tint[index]))
        for index, channel in enumerate(rgb)
    )


def sample_overhead_light_position(rng: random.Random, randomize: bool) -> tuple[float, float, float]:
    if not randomize:
        return (TABLE_CENTER_X, TABLE_CENTER_Y, OVERHEAD_LIGHT_Z)

    light_x = rng.uniform(
        TABLE_CENTER_X - TABLE_TOP_HALF_X,
        TABLE_CENTER_X + TABLE_TOP_HALF_X,
    )
    light_y = rng.uniform(
        TABLE_CENTER_Y - TABLE_TOP_HALF_Y,
        TABLE_CENTER_Y + TABLE_TOP_HALF_Y,
    )
    light_z = rng.uniform(1.70, 2.30)
    return (light_x, light_y, light_z)


def overhead_light_spec(rng: random.Random, randomize: bool) -> dict[str, str]:
    position = sample_overhead_light_position(rng, randomize)
    diffuse = (0.95, 0.95, 0.92)
    specular = (0.20, 0.20, 0.20)
    if randomize:
        diffuse = scale_rgb(diffuse, rng, intensity_scale_range=(0.80, 1.10), tint_range=0.05)
        specular = scale_rgb(specular, rng, intensity_scale_range=(0.70, 1.20), tint_range=0.03)

    return {
        "name": "table_overhead_light",
        "pos": format_vec3(position),
        "dir": format_vec3(VERTICAL_DIR),
        "directional": "true",
        "diffuse": format_vec3(diffuse),
        "specular": format_vec3(specular),
        "castshadow": "true",
        "active": "true",
    }


def configure_headlight(visual: ET.Element, randomize: bool, rng: random.Random) -> None:
    headlight = visual.find("headlight")
    if headlight is None:
        headlight = ET.SubElement(visual, "headlight")

    # Keep ambient low so the overhead vertical light dominates on the table.
    ambient = (0.12, 0.12, 0.14)
    diffuse = (0.18, 0.18, 0.20)
    specular = (0.02, 0.02, 0.02)
    if randomize:
        ambient = scale_rgb(ambient, rng, intensity_scale_range=(0.85, 1.15), tint_range=0.02)
        diffuse = scale_rgb(diffuse, rng, intensity_scale_range=(0.85, 1.15), tint_range=0.02)

    headlight.set("ambient", format_vec3(ambient))
    headlight.set("diffuse", format_vec3(diffuse))
    headlight.set("specular", format_vec3(specular))


def remove_managed_lights(worldbody: ET.Element) -> None:
    for light in list(worldbody.findall("light")):
        light_name = light.get("name")
        if light_name in MANAGED_LIGHT_NAMES or light_name is None:
            worldbody.remove(light)


def add_overhead_light(worldbody: ET.Element, light_spec: dict[str, str]) -> None:
    worldbody.insert(0, ET.Element("light", light_spec))


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    input_scene = Path(args.input_scene)
    output_scene = Path(args.output_scene)
    output_scene.parent.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(input_scene)
    root = tree.getroot()
    rebase_include_paths(root, input_scene)

    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    configure_headlight(visual, args.randomize, rng)

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"Missing worldbody in scene XML: {input_scene}")

    remove_managed_lights(worldbody)
    light_spec = overhead_light_spec(rng, args.randomize)
    add_overhead_light(worldbody, light_spec)

    tree.write(output_scene, encoding="utf-8", xml_declaration=False)

    print(
        "[configure_scene_43dof_lighting] "
        f"output={output_scene} randomize={args.randomize} "
        f"table_overhead_light pos=({light_spec['pos']}) "
        f"dir=({light_spec['dir']}) diffuse=({light_spec['diffuse']})"
    )


if __name__ == "__main__":
    main()
