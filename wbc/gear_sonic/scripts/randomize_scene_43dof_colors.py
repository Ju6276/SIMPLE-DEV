"""Create a copy of scene_43dof.xml with randomized object colors."""

from __future__ import annotations

import argparse
import colorsys
import random
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-scene", required=True, help="Source MuJoCo scene XML path")
    parser.add_argument("--output-scene", required=True, help="Output MuJoCo scene XML path")
    parser.add_argument(
        "--table-only",
        action="store_true",
        help="Randomize only the table color and keep the cup color unchanged.",
    )
    parser.add_argument(
        "--min-rgb-distance",
        type=float,
        default=0.20,
        help="Minimum Euclidean distance between cup and table RGB values.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible color sampling.",
    )
    return parser.parse_args()


def rebase_include_paths(root: ET.Element, input_scene: Path) -> None:
    for include_node in root.findall("include"):
        include_file = include_node.get("file")
        if include_file and not Path(include_file).is_absolute():
            include_node.set("file", str((input_scene.parent / include_file).resolve()))


def random_rgba(rng: random.Random) -> list[float]:
    hue = rng.random()
    saturation = rng.uniform(0.40, 0.95)
    value = rng.uniform(0.45, 0.95)
    red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
    return [red, green, blue, 1.0]


def parse_rgba(raw_rgba: str) -> list[float]:
    return [float(value) for value in raw_rgba.split()]


def rgb_distance(left: list[float], right: list[float]) -> float:
    return sum((left[i] - right[i]) ** 2 for i in range(3)) ** 0.5


def format_rgba(rgba: list[float]) -> str:
    return f"{rgba[0]:.6f} {rgba[1]:.6f} {rgba[2]:.6f} {rgba[3]:.6f}"


def sample_distinct_colors(
    rng: random.Random,
    min_rgb_distance: float,
    max_attempts: int = 100,
) -> tuple[list[float], list[float]]:
    cup_rgba = random_rgba(rng)
    for _ in range(max_attempts):
        table_rgba = random_rgba(rng)
        if rgb_distance(cup_rgba, table_rgba) >= min_rgb_distance:
            return cup_rgba, table_rgba

    table_rgba = cup_rgba.copy()
    table_rgba[0] = 1.0 - table_rgba[0]
    table_rgba[1] = 1.0 - table_rgba[1]
    table_rgba[2] = 1.0 - table_rgba[2]
    return cup_rgba, table_rgba


def sample_table_color_distinct_from_cup(
    rng: random.Random,
    cup_rgba: list[float],
    min_rgb_distance: float,
    max_attempts: int = 100,
) -> list[float]:
    for _ in range(max_attempts):
        table_rgba = random_rgba(rng)
        if rgb_distance(cup_rgba, table_rgba) >= min_rgb_distance:
            return table_rgba

    table_rgba = cup_rgba.copy()
    table_rgba[0] = 1.0 - table_rgba[0]
    table_rgba[1] = 1.0 - table_rgba[1]
    table_rgba[2] = 1.0 - table_rgba[2]
    return table_rgba


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    input_scene = Path(args.input_scene)
    output_scene = Path(args.output_scene)
    output_scene.parent.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(input_scene)
    root = tree.getroot()
    rebase_include_paths(root, input_scene)

    asset = root.find("asset")
    if asset is None:
        raise ValueError(f"Missing asset section in scene XML: {input_scene}")

    table_mat = asset.find("material[@name='table_mat']")
    cup_mat = asset.find("material[@name='cup_mat']")
    if table_mat is None or cup_mat is None:
        raise ValueError(f"Missing table_mat or cup_mat in scene XML: {input_scene}")

    if args.table_only:
        cup_rgba = parse_rgba(cup_mat.attrib["rgba"])
        table_rgba = sample_table_color_distinct_from_cup(
            rng,
            cup_rgba,
            args.min_rgb_distance,
        )
        table_mat.set("rgba", format_rgba(table_rgba))
    else:
        cup_rgba, table_rgba = sample_distinct_colors(rng, args.min_rgb_distance)
        cup_mat.set("rgba", format_rgba(cup_rgba))
        table_mat.set("rgba", format_rgba(table_rgba))

    tree.write(output_scene, encoding="utf-8", xml_declaration=False)

    print(
        "[randomize_scene_43dof_colors] "
        f"output={output_scene} table_only={args.table_only} "
        f"cup_rgba={cup_mat.attrib['rgba']} "
        f"table_rgba={table_mat.attrib['rgba']}"
    )


if __name__ == "__main__":
    main()
