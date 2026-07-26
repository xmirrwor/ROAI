#!/usr/bin/env python3
"""Generate the no-head MiniDuck URDF from the archived MuJoCo XML."""

from __future__ import annotations

import math
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parent
SOURCE_XML = ROOT / "MINIDUCK_URDF" / "_archive" / "xmls" / "open_duck_mini_v2_no_head.xml"
OUTPUTS = [
    ROOT / "urdf" / "MINIDUCK.urdf",
    ROOT / "MINIDUCK_URDF" / "MINIDUCK.urdf",
]
ZIP_PATH = ROOT / "MINIDUCK_URDF.zip"

DEFAULT_ROOT_INERTIA = {
    "mass": "0.001",
    "ixx": "0.000001",
    "ixy": "0",
    "ixz": "0",
    "iyy": "0.000001",
    "iyz": "0",
    "izz": "0.000001",
}
DEFAULT_VELOCITY_LIMIT = 5.24
SKIPPED_MESHES = {"left_cache", "right_cache"}
SKIPPED_MATERIALS = {"left_cache_material", "right_cache_material"}


def parse_floats(raw: str | None, default: list[float]) -> list[float]:
    if raw is None:
        return list(default)
    return [float(item) for item in raw.split()]


def fmt(value: float) -> str:
    if abs(value) < 1e-12:
        value = 0.0
    return f"{value:.12g}"


def fmt_vec(values: list[float]) -> str:
    return " ".join(fmt(v) for v in values)


def quat_to_rpy_wxyz(quat_wxyz: list[float]) -> list[float]:
    w, x, y, z = quat_wxyz
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0.0:
        return [0.0, 0.0, 0.0]
    w /= norm
    x /= norm
    y /= norm
    z /= norm

    rot = [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ]

    # URDF stores roll-pitch-yaw. Several Open Duck body frames sit exactly on
    # pitch singularities, so compute from the rotation matrix and choose a
    # stable equivalent representation instead of letting atan2(0, 0) drift.
    eps = 1e-9
    if -1.0 + eps < rot[2][0] < 1.0 - eps:
        pitch = math.asin(-rot[2][0])
        roll = math.atan2(rot[2][1], rot[2][2])
        yaw = math.atan2(rot[1][0], rot[0][0])
    elif rot[2][0] <= -1.0 + eps:
        pitch = math.pi / 2.0
        yaw = 0.0
        roll = math.atan2(rot[0][1], rot[1][1])
    else:
        pitch = -math.pi / 2.0
        yaw = 0.0
        roll = math.atan2(-rot[0][1], rot[1][1])

    return [roll, pitch, yaw]


def indent(elem: ET.Element, level: int = 0) -> None:
    newline = "\n" + "  " * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = newline + "  "
        for child in elem:
            indent(child, level + 1)
        if not elem[-1].tail or not elem[-1].tail.strip():
            elem[-1].tail = newline
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = newline


def build_default_maps(root: ET.Element) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    joint_defaults: dict[str, dict[str, str]] = {}
    actuator_defaults: dict[str, dict[str, str]] = {}

    def walk(node: ET.Element, inherited_joint: dict[str, str], inherited_position: dict[str, str]) -> None:
        joint_cfg = dict(inherited_joint)
        joint_elem = node.find("joint")
        if joint_elem is not None:
            joint_cfg.update(joint_elem.attrib)

        position_cfg = dict(inherited_position)
        position_elem = node.find("position")
        if position_elem is not None:
            position_cfg.update(position_elem.attrib)

        class_name = node.attrib.get("class")
        if class_name:
            joint_defaults[class_name] = dict(joint_cfg)
            actuator_defaults[class_name] = dict(position_cfg)

        for child in node.findall("default"):
            walk(child, joint_cfg, position_cfg)

    for default_node in root.findall("default"):
        walk(default_node, {}, {})

    return joint_defaults, actuator_defaults


def build_mesh_map(root: ET.Element) -> dict[str, str]:
    asset = root.find("asset")
    if asset is None:
        return {}

    meshes: dict[str, str] = {}
    for mesh in asset.findall("mesh"):
        file_name = mesh.attrib["file"]
        mesh_name = mesh.attrib.get("name", Path(file_name).stem)
        meshes[mesh_name] = file_name
    return meshes


def build_materials(root: ET.Element, robot: ET.Element) -> None:
    asset = root.find("asset")
    if asset is None:
        return

    for material in asset.findall("material"):
        if material.attrib["name"] in SKIPPED_MATERIALS:
            continue
        mat_elem = ET.SubElement(robot, "material", name=material.attrib["name"])
        ET.SubElement(mat_elem, "color", rgba=material.attrib["rgba"])


def inertial_to_urdf(link: ET.Element, body: ET.Element) -> None:
    inertial = body.find("inertial")
    inertial_elem = ET.SubElement(link, "inertial")
    ET.SubElement(
        inertial_elem,
        "origin",
        xyz=fmt_vec(parse_floats(inertial.attrib.get("pos") if inertial is not None else None, [0.0, 0.0, 0.0])),
        rpy="0 0 0",
    )

    if inertial is None:
        ET.SubElement(inertial_elem, "mass", value=DEFAULT_ROOT_INERTIA["mass"])
        ET.SubElement(
            inertial_elem,
            "inertia",
            ixx=DEFAULT_ROOT_INERTIA["ixx"],
            ixy=DEFAULT_ROOT_INERTIA["ixy"],
            ixz=DEFAULT_ROOT_INERTIA["ixz"],
            iyy=DEFAULT_ROOT_INERTIA["iyy"],
            iyz=DEFAULT_ROOT_INERTIA["iyz"],
            izz=DEFAULT_ROOT_INERTIA["izz"],
        )
        return

    ixx, iyy, izz, ixy, ixz, iyz = parse_floats(inertial.attrib["fullinertia"], [0.0] * 6)
    ET.SubElement(inertial_elem, "mass", value=fmt(float(inertial.attrib["mass"])))
    ET.SubElement(
        inertial_elem,
        "inertia",
        ixx=fmt(ixx),
        ixy=fmt(ixy),
        ixz=fmt(ixz),
        iyy=fmt(iyy),
        iyz=fmt(iyz),
        izz=fmt(izz),
    )


def geom_origin_attrs(geom: ET.Element) -> dict[str, str]:
    pos = parse_floats(geom.attrib.get("pos"), [0.0, 0.0, 0.0])
    quat = parse_floats(geom.attrib.get("quat"), [1.0, 0.0, 0.0, 0.0])
    return {"xyz": fmt_vec(pos), "rpy": fmt_vec(quat_to_rpy_wxyz(quat))}


def joint_origin_attrs(body: ET.Element) -> dict[str, str]:
    pos = parse_floats(body.attrib.get("pos"), [0.0, 0.0, 0.0])
    quat = parse_floats(body.attrib.get("quat"), [1.0, 0.0, 0.0, 0.0])
    return {"xyz": fmt_vec(pos), "rpy": fmt_vec(quat_to_rpy_wxyz(quat))}


def add_geometry(container: ET.Element, mesh_file: str, origin_attrs: dict[str, str], material_name: str | None = None) -> None:
    ET.SubElement(container, "origin", **origin_attrs)
    geometry = ET.SubElement(container, "geometry")
    ET.SubElement(geometry, "mesh", filename=f"../meshes/{mesh_file}")
    if material_name is not None:
        ET.SubElement(container, "material", name=material_name)


def build_link(robot: ET.Element, body: ET.Element, mesh_map: dict[str, str]) -> None:
    link = ET.SubElement(robot, "link", name=body.attrib["name"])
    inertial_to_urdf(link, body)

    direct_geoms = [
        child
        for child in body
        if (
            child.tag == "geom"
            and child.attrib.get("type", "mesh") == "mesh"
            and child.attrib.get("mesh") not in SKIPPED_MESHES
        )
    ]
    explicit_collisions = [geom for geom in direct_geoms if geom.attrib.get("class") == "collision"]
    collision_geoms = explicit_collisions

    for geom in direct_geoms:
        if geom.attrib.get("class") == "collision":
            continue
        visual = ET.SubElement(link, "visual")
        mesh_file = mesh_map[geom.attrib["mesh"]]
        add_geometry(visual, mesh_file, geom_origin_attrs(geom), geom.attrib.get("material"))

    for geom in collision_geoms:
        collision = ET.SubElement(link, "collision")
        mesh_file = mesh_map[geom.attrib["mesh"]]
        add_geometry(collision, mesh_file, geom_origin_attrs(geom))


def build_joint(
    robot: ET.Element,
    parent_name: str,
    body: ET.Element,
    joint_defaults: dict[str, dict[str, str]],
    actuator_defaults: dict[str, dict[str, str]],
    actuator_limits: dict[str, float],
) -> None:
    mjcf_joints = [
        child
        for child in body
        if child.tag == "joint" and child.attrib.get("type") != "free" and child.attrib.get("name") != "floating_base"
    ]

    if not mjcf_joints:
        joint = ET.SubElement(robot, "joint", name=f"{parent_name}_to_{body.attrib['name']}", type="fixed")
        ET.SubElement(joint, "origin", **joint_origin_attrs(body))
        ET.SubElement(joint, "parent", link=parent_name)
        ET.SubElement(joint, "child", link=body.attrib["name"])
        return

    if len(mjcf_joints) != 1:
        raise ValueError(f"Expected one actuated joint in body {body.attrib['name']}, got {len(mjcf_joints)}")

    mjcf_joint = mjcf_joints[0]
    joint_type = {"hinge": "revolute", "slide": "prismatic"}.get(mjcf_joint.attrib.get("type", "hinge"), "revolute")
    joint = ET.SubElement(robot, "joint", name=mjcf_joint.attrib["name"], type=joint_type)
    ET.SubElement(joint, "origin", **joint_origin_attrs(body))
    ET.SubElement(joint, "parent", link=parent_name)
    ET.SubElement(joint, "child", link=body.attrib["name"])

    axis = parse_floats(mjcf_joint.attrib.get("axis"), [0.0, 0.0, 1.0])
    ET.SubElement(joint, "axis", xyz=fmt_vec(axis))

    limits = parse_floats(mjcf_joint.attrib.get("range"), [0.0, 0.0])
    effort = actuator_limits.get(mjcf_joint.attrib["name"], 3.23)
    velocity = float(mjcf_joint.attrib.get("velocity", DEFAULT_VELOCITY_LIMIT))
    ET.SubElement(
        joint,
        "limit",
        lower=fmt(limits[0]),
        upper=fmt(limits[1]),
        effort=fmt(effort),
        velocity=fmt(velocity),
    )

    class_name = mjcf_joint.attrib.get("class")
    resolved_joint_defaults = joint_defaults.get(class_name, {})
    resolved_actuator_defaults = actuator_defaults.get(class_name, {})
    damping = mjcf_joint.attrib.get("damping", resolved_joint_defaults.get("damping"))
    friction = mjcf_joint.attrib.get("frictionloss", resolved_joint_defaults.get("frictionloss"))

    dynamics_attrs = {}
    if damping is not None:
        dynamics_attrs["damping"] = fmt(float(damping))
    if friction is not None:
        dynamics_attrs["friction"] = fmt(float(friction))
    if dynamics_attrs:
        ET.SubElement(joint, "dynamics", **dynamics_attrs)

    # Keep the per-class actuator defaults available for future tuning/debugging.
    if resolved_actuator_defaults.get("kp") is not None:
        joint.append(ET.Comment(f" MuJoCo position kp={resolved_actuator_defaults['kp']} "))


def collect_actuator_limits(root: ET.Element, actuator_defaults: dict[str, dict[str, str]]) -> dict[str, float]:
    limits: dict[str, float] = {}
    actuator = root.find("actuator")
    if actuator is None:
        return limits

    for position in actuator.findall("position"):
        joint_name = position.attrib.get("joint")
        class_name = position.attrib.get("class")
        default_cfg = actuator_defaults.get(class_name, {})
        forcerange = position.attrib.get("forcerange", default_cfg.get("forcerange"))
        if joint_name is None or forcerange is None:
            continue
        force_min, force_max = parse_floats(forcerange, [-3.23, 3.23])
        limits[joint_name] = max(abs(force_min), abs(force_max))
    return limits


def walk_bodies(
    robot: ET.Element,
    parent_name: str,
    parent_body: ET.Element,
    mesh_map: dict[str, str],
    joint_defaults: dict[str, dict[str, str]],
    actuator_defaults: dict[str, dict[str, str]],
    actuator_limits: dict[str, float],
) -> None:
    for child in parent_body.findall("body"):
        build_link(robot, child, mesh_map)
        build_joint(robot, parent_name, child, joint_defaults, actuator_defaults, actuator_limits)
        walk_bodies(robot, child.attrib["name"], child, mesh_map, joint_defaults, actuator_defaults, actuator_limits)


def write_urdf(urdf_root: ET.Element) -> None:
    indent(urdf_root)
    xml_bytes = ET.tostring(urdf_root, encoding="utf-8", xml_declaration=True)
    for output in OUTPUTS:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(xml_bytes)


def rebuild_zip() -> None:
    with ZipFile(ZIP_PATH, "w", compression=ZIP_DEFLATED) as archive:
        for path in sorted((ROOT / "MINIDUCK_URDF").rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(ROOT))


def main() -> None:
    mjcf_root = ET.parse(SOURCE_XML).getroot()
    worldbody = mjcf_root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"Missing worldbody in {SOURCE_XML}")

    base_body = worldbody.find("body")
    if base_body is None or base_body.attrib.get("name") != "base":
        raise ValueError(f"Expected root body 'base' in {SOURCE_XML}")

    mesh_map = build_mesh_map(mjcf_root)
    joint_defaults, actuator_defaults = build_default_maps(mjcf_root)
    actuator_limits = collect_actuator_limits(mjcf_root, actuator_defaults)

    robot = ET.Element("robot", name="MINIDUCK_URDF")
    mujoco = ET.SubElement(robot, "mujoco")
    ET.SubElement(mujoco, "compiler", meshdir="../meshes", discardvisual="false")
    robot.append(
        ET.Comment(
            "Converted from the no-head Open Duck Mini V2 MuJoCo XML. "
            "Sensors, actuators, and MuJoCo-only solver options are omitted."
        )
    )

    build_materials(mjcf_root, robot)
    build_link(robot, base_body, mesh_map={})
    walk_bodies(robot, base_body.attrib["name"], base_body, mesh_map, joint_defaults, actuator_defaults, actuator_limits)
    write_urdf(robot)
    rebuild_zip()


if __name__ == "__main__":
    main()
