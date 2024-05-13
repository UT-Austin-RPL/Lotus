from robosuite.models.arenas import Arena
from robosuite.utils.mjcf_utils import xml_path_completion

from lotus.libero.envs.arenas.style import get_texture_filename


class CoffeeTableArena(Arena):
    """Empty workspace."""

    def __init__(
        self,
        xml="arenas/table_arena.xml",
        floor_style="light-gray",
        wall_style="light-gray-plaster",
    ):
        super().__init__(xml_path_completion(xml))
        self.coffee_table_body = self.worldbody.find("./body[@name='coffee_table']")

        texplane = self.asset.find("./texture[@name='texplane']")
        plane_file = texplane.get("file")
        plane_file = "/".join(
            plane_file.split("/")[:-1]
            + [get_texture_filename(type="floor", style=floor_style)]
        )
        texplane.set("file", plane_file)

        texwall = self.asset.find("./texture[@name='tex-wall']")
        wall_file = texwall.get("file")
        wall_file = "/".join(
            wall_file.split("/")[:-1]
            + [get_texture_filename(type="wall", style=wall_style)]
        )
        texwall.set("file", wall_file)
