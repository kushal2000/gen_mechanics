"""collision_from_visuals: collisions rebuilt from visual meshes."""

from __future__ import annotations

import xml.etree.ElementTree as ET

from isaacsimenvs.inhand_reorient.collision_from_visuals import collisions_from_visuals

_URDF = """<robot name="r">
  <link name="a">
    <visual><origin xyz="0 0 0.1"/><geometry><mesh filename="/m/a.obj" scale="1 1 1"/></geometry></visual>
    <collision><geometry><box size="0.1 0.1 0.1"/></geometry></collision>
  </link>
  <link name="b">
    <collision><geometry><sphere radius="0.01"/></geometry></collision>
  </link>
  <link name="c">
    <visual><geometry><box size="1 1 1"/></geometry></visual>
    <collision><geometry><box size="1 1 1"/></geometry></collision>
  </link>
</robot>"""


def test_mesh_visuals_replace_collisions_and_other_links_are_kept(tmp_path):
    src, dst = tmp_path / "in.urdf", tmp_path / "out.urdf"
    src.write_text(_URDF)
    assert collisions_from_visuals(src, dst) == 1
    links = {l.get("name"): l for l in ET.parse(dst).getroot().findall("link")}
    cols = links["a"].findall("collision")
    assert len(cols) == 1
    assert cols[0].find("geometry/mesh").get("filename") == "/m/a.obj"
    assert cols[0].find("origin").get("xyz") == "0 0 0.1"
    assert links["b"].find("collision/geometry/sphere") is not None
    assert links["c"].find("collision/geometry/box") is not None
