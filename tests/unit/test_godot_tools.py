import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

import unittest
from unittest.mock import patch, mock_open
from src.aura.tools.godot_tools import read_godot_scene_tree

class TestGodotTools(unittest.TestCase):

    tscn_content = """
[gd_scene load_steps=2 format=3 uid="uid://c6804g7bsbc2w"]

[ext_resource type="Script" path="res://player.gd" id="1_t0g0g"]

[node name="Player" type="CharacterBody2D"]
script = ExtResource("1_t0g0g")

[node name="CollisionShape2D" type="CollisionShape2D" parent="."]
shape = SubResource("RectangleShape2D_t0g0g")

[node name="Camera2D" type="Camera2D" parent="."]
zoom = Vector2(2, 2)

[node name="Gun" type="Node2D" parent="."]

[node name="Muzzle" type="Node2D" parent="Gun"]
position = Vector2(0, -30)
    """

    def test_read_godot_scene_tree(self):
        with patch("pathlib.Path.read_text", return_value=self.tscn_content) as mock_read_text:
            result = read_godot_scene_tree("dummy_path.tscn")

            self.assertTrue(result.get("success"))
            scene_tree = result.get("scene_tree")
            self.assertEqual(scene_tree["root"]["name"], "Player")

            player_node = scene_tree["root"]
            self.assertEqual(player_node["type"], "CharacterBody2D")
            self.assertEqual(player_node["parent_path"], ".")
            self.assertEqual(len(player_node["children"]), 3)

            gun_node = next((child for child in player_node["children"] if child["name"] == "Gun"), None)
            self.assertIsNotNone(gun_node)
            self.assertEqual(gun_node["type"], "Node2D")
            self.assertEqual(len(gun_node["children"]), 1)

            muzzle_node = gun_node["children"][0]
            self.assertEqual(muzzle_node["name"], "Muzzle")
            self.assertEqual(muzzle_node["type"], "Node2D")

if __name__ == '__main__':
    unittest.main()