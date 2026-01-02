"""
Blender Scene Builder
Builds 3D scenes from scene graph JSON data
"""

import bpy
import math
from mathutils import Vector, Euler
from typing import Dict, Any, List, Optional


class SceneBuilder:
    """
    Build Blender scenes from scene graph JSON
    """

    def __init__(self, scene_graph: Dict[str, Any], collection_name: str = "Generated Scene"):
        """
        Initialize scene builder

        Args:
            scene_graph: Scene graph dictionary from AI worker
            collection_name: Name for the generated collection
        """
        self.scene_graph = scene_graph
        self.collection_name = collection_name
        self.collection = None
        self.objects_map = {}  # Map object IDs to Blender objects
        self.materials_map = {}  # Map material names to Blender materials

    def build_scene(self, clear_existing: bool = True) -> Dict[str, Any]:
        """
        Build complete scene from scene graph

        Args:
            clear_existing: Clear existing scene objects

        Returns:
            Dictionary with build results
        """
        results = {
            "success": False,
            "camera": None,
            "objects_created": 0,
            "lights_created": 0,
            "materials_created": 0,
            "errors": []
        }

        try:
            # Create or get collection
            self.collection = self._get_or_create_collection(self.collection_name)

            if clear_existing:
                self._clear_collection(self.collection)

            # Build in order
            results["camera"] = self.build_camera()
            self.build_room()
            results["objects_created"] = self.build_objects()
            results["lights_created"] = self.build_lighting()
            results["materials_created"] = self.apply_materials()

            results["success"] = True

        except Exception as e:
            results["errors"].append(str(e))
            print(f"Scene build failed: {e}")

        return results

    def build_camera(self) -> Optional[bpy.types.Object]:
        """
        Create camera from scene graph

        Returns:
            Camera object
        """
        camera_data = self.scene_graph.get("camera")
        if not camera_data:
            return None

        # Create camera
        cam_data = bpy.data.cameras.new(name="GeneratedCamera")
        cam_obj = bpy.data.objects.new("GeneratedCamera", cam_data)

        # Set FOV
        fov_deg = camera_data.get("fov_deg", 50.0)
        cam_data.lens_unit = 'FOV'
        cam_data.angle = math.radians(fov_deg)

        # Set position
        position = camera_data.get("position_hint", [0, -4, 1.5])
        cam_obj.location = Vector(position)

        # Set rotation (pitch, yaw, roll)
        pitch = math.radians(camera_data.get("pitch_deg", 0))
        yaw = math.radians(camera_data.get("yaw_deg", 0))
        roll = math.radians(camera_data.get("roll_deg", 0))

        # Convert to Blender's rotation order (XYZ Euler)
        # Blender: X=pitch, Z=yaw, Y=roll
        cam_obj.rotation_euler = Euler((pitch + math.radians(90), roll, yaw), 'XYZ')

        # Link to collection
        self.collection.objects.link(cam_obj)

        # Set as active camera
        bpy.context.scene.camera = cam_obj

        print(f"✓ Camera created: FOV={fov_deg}°, pos={position}")
        return cam_obj

    def build_room(self) -> List[bpy.types.Object]:
        """
        Build room geometry (floor, walls, ceiling)

        Returns:
            List of created mesh objects
        """
        room_data = self.scene_graph.get("room")
        if not room_data:
            return []

        created_objects = []
        planes = room_data.get("planes", [])

        for plane in planes:
            obj = self._create_plane(plane, room_data.get("dimensions_m", [4, 4, 2.6]))
            if obj:
                created_objects.append(obj)
                self.collection.objects.link(obj)

        # Create window if specified
        window_data = room_data.get("window")
        if window_data:
            window_obj = self._create_window(window_data)
            if window_obj:
                created_objects.append(window_obj)
                self.collection.objects.link(window_obj)

        print(f"✓ Room created: {len(created_objects)} planes")
        return created_objects

    def _create_plane(self, plane: Dict[str, Any], room_dims: List[float]) -> Optional[bpy.types.Object]:
        """
        Create a single plane (floor, wall, ceiling)

        Args:
            plane: Plane definition
            room_dims: Room dimensions [width, depth, height]

        Returns:
            Plane mesh object
        """
        name = plane.get("name", "plane")
        normal = Vector(plane.get("normal", [0, 0, 1]))
        distance = plane.get("distance", 0.0)

        # Create mesh
        bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 0, 0))
        obj = bpy.context.active_object
        obj.name = f"Room_{name}"

        # Determine size and position based on plane type
        if "floor" in name.lower():
            obj.scale = (room_dims[0] / 2, room_dims[1] / 2, 1)
            obj.location = (0, 0, 0)

        elif "ceiling" in name.lower():
            obj.scale = (room_dims[0] / 2, room_dims[1] / 2, 1)
            obj.location = (0, 0, room_dims[2])

        elif "back" in name.lower():
            obj.scale = (room_dims[0] / 2, room_dims[2] / 2, 1)
            obj.location = (0, distance, room_dims[2] / 2)
            obj.rotation_euler = (math.radians(90), 0, 0)

        elif "left" in name.lower():
            obj.scale = (room_dims[1] / 2, room_dims[2] / 2, 1)
            obj.location = (-distance, 0, room_dims[2] / 2)
            obj.rotation_euler = (math.radians(90), 0, math.radians(90))

        elif "right" in name.lower():
            obj.scale = (room_dims[1] / 2, room_dims[2] / 2, 1)
            obj.location = (distance, 0, room_dims[2] / 2)
            obj.rotation_euler = (math.radians(90), 0, math.radians(-90))

        return obj

    def _create_window(self, window_data: Dict[str, Any]) -> Optional[bpy.types.Object]:
        """
        Create window plane

        Args:
            window_data: Window specification

        Returns:
            Window mesh object
        """
        rect = window_data.get("rect_world", {})
        x = rect.get("x", 0)
        y = rect.get("y", 4)
        width = rect.get("width", 1.5)
        height = rect.get("height", 1.4)

        # Create plane for window
        bpy.ops.mesh.primitive_plane_add(size=1)
        obj = bpy.context.active_object
        obj.name = "Room_Window"

        obj.scale = (width / 2, height / 2, 1)
        obj.location = (x, y, height / 2 + 1.2)  # Approximate window height
        obj.rotation_euler = (math.radians(90), 0, 0)

        return obj

    def build_objects(self) -> int:
        """
        Build all scene objects

        Returns:
            Number of objects created
        """
        objects_data = self.scene_graph.get("objects", [])
        count = 0

        for obj_data in objects_data:
            obj = self._create_object(obj_data)
            if obj:
                self.collection.objects.link(obj)
                self.objects_map[obj_data["id"]] = obj
                count += 1

        print(f"✓ Objects created: {count}")
        return count

    def _create_object(self, obj_data: Dict[str, Any]) -> Optional[bpy.types.Object]:
        """
        Create a single object based on proxy type

        Args:
            obj_data: Object specification

        Returns:
            Created Blender object
        """
        proxy_type = obj_data.get("proxy_type", "box_subdiv")
        obj_id = obj_data.get("id", "object")
        obj_type = obj_data.get("type", "object")

        # Create geometry based on proxy type
        if proxy_type == "box_subdiv":
            obj = self._create_box_proxy(obj_data)
        elif proxy_type == "card_cutout":
            obj = self._create_card_proxy(obj_data)
        elif proxy_type == "lowpoly_mesh":
            obj = self._create_lowpoly_proxy(obj_data)
        else:
            obj = self._create_box_proxy(obj_data)  # Fallback

        if obj:
            obj.name = f"{obj_type}_{obj_id}"

            # Set transform
            position = obj_data.get("world_position", [0, 0, 0])
            rotation = obj_data.get("world_rotation", [0, 0, 0])
            scale = obj_data.get("world_scale", [1, 1, 1])

            obj.location = Vector(position)
            obj.rotation_euler = Euler([math.radians(r) for r in rotation], 'XYZ')
            obj.scale = Vector(scale)

        return obj

    def _create_box_proxy(self, obj_data: Dict[str, Any]) -> bpy.types.Object:
        """Create subdivided box proxy"""
        bpy.ops.mesh.primitive_cube_add()
        obj = bpy.context.active_object

        # Add subdivision modifier
        subdiv = obj.modifiers.new(name="Subdivision", type='SUBSURF')
        subdiv.levels = 1
        subdiv.render_levels = 2

        return obj

    def _create_card_proxy(self, obj_data: Dict[str, Any]) -> bpy.types.Object:
        """Create card cutout proxy (for people, animals)"""
        bpy.ops.mesh.primitive_plane_add()
        obj = bpy.context.active_object

        # Make it face the camera
        obj.rotation_euler = (0, 0, 0)

        return obj

    def _create_lowpoly_proxy(self, obj_data: Dict[str, Any]) -> bpy.types.Object:
        """Create low-poly mesh proxy"""
        obj_type = obj_data.get("type", "object")

        # Simple proxy based on object type
        if obj_type in ["plant", "tree"]:
            # Cone for plants
            bpy.ops.mesh.primitive_cone_add(vertices=6)
        elif obj_type in ["lamp", "light"]:
            # Cylinder for lamps
            bpy.ops.mesh.primitive_cylinder_add(vertices=8)
        else:
            # Default to UV sphere
            bpy.ops.mesh.primitive_uv_sphere_add(segments=8, ring_count=6)

        obj = bpy.context.active_object
        return obj

    def build_lighting(self) -> int:
        """
        Build lighting setup

        Returns:
            Number of lights created
        """
        lighting_data = self.scene_graph.get("lighting", {})
        count = 0

        # Key light
        if "key_light" in lighting_data:
            light = self._create_light("Key", lighting_data["key_light"])
            if light:
                self.collection.objects.link(light)
                count += 1

        # Fill light
        if "fill_light" in lighting_data:
            light = self._create_light("Fill", lighting_data["fill_light"])
            if light:
                self.collection.objects.link(light)
                count += 1

        # Ambient (world lighting)
        if "ambient" in lighting_data:
            self._setup_world_lighting(lighting_data["ambient"])

        print(f"✓ Lights created: {count}")
        return count

    def _create_light(self, name: str, light_data: Dict[str, Any]) -> Optional[bpy.types.Object]:
        """
        Create a single light

        Args:
            name: Light name
            light_data: Light specification

        Returns:
            Light object
        """
        light_type = light_data.get("type", "area").upper()

        # Create light data
        light_data_obj = bpy.data.lights.new(name=f"Light_{name}", type='AREA')

        # Set properties
        strength = light_data.get("strength", 100.0)
        temp_kelvin = light_data.get("temp_kelvin", 6500)
        size = light_data.get("size", 0.5)

        light_data_obj.energy = strength
        light_data_obj.size = size

        # Convert color temperature to RGB (simplified)
        color = self._kelvin_to_rgb(temp_kelvin)
        light_data_obj.color = color

        # Create light object
        light_obj = bpy.data.objects.new(name=f"Light_{name}", object_data=light_data_obj)

        # Set position and rotation
        position = light_data.get("position", [0, 0, 3])
        rotation = light_data.get("rotation", [45, 0, 0])

        light_obj.location = Vector(position)
        light_obj.rotation_euler = Euler([math.radians(r) for r in rotation], 'XYZ')

        return light_obj

    def _kelvin_to_rgb(self, kelvin: float) -> tuple:
        """
        Convert color temperature to RGB (simplified approximation)

        Args:
            kelvin: Color temperature in Kelvin

        Returns:
            RGB tuple (0-1 range)
        """
        # Simplified conversion
        if kelvin < 3000:
            return (1.0, 0.6, 0.3)  # Warm
        elif kelvin < 5000:
            return (1.0, 0.9, 0.7)  # Neutral warm
        elif kelvin < 7000:
            return (1.0, 1.0, 1.0)  # Neutral
        else:
            return (0.8, 0.9, 1.0)  # Cool

    def _setup_world_lighting(self, ambient_data: Dict[str, Any]):
        """
        Setup world/ambient lighting

        Args:
            ambient_data: Ambient light specification
        """
        world = bpy.context.scene.world
        if not world:
            world = bpy.data.worlds.new("GeneratedWorld")
            bpy.context.scene.world = world

        # Use nodes for world
        world.use_nodes = True
        nodes = world.node_tree.nodes
        nodes.clear()

        # Add background node
        bg_node = nodes.new(type='ShaderNodeBackground')
        output_node = nodes.new(type='ShaderNodeOutputWorld')

        # Set strength
        strength = ambient_data.get("strength", 0.3)
        bg_node.inputs['Strength'].default_value = strength

        # Set color based on temperature
        temp_kelvin = ambient_data.get("temp_kelvin", 5000)
        bg_node.inputs['Color'].default_value = (*self._kelvin_to_rgb(temp_kelvin), 1.0)

        # Link nodes
        world.node_tree.links.new(bg_node.outputs['Background'], output_node.inputs['Surface'])

    def apply_materials(self) -> int:
        """
        Apply materials to objects

        Returns:
            Number of materials created
        """
        materials_data = self.scene_graph.get("materials", [])
        count = 0

        for mat_data in materials_data:
            target_id = mat_data.get("target_id")

            # Handle special cases (floor, walls, ceiling)
            if target_id in ["floor", "walls", "ceiling"]:
                self._apply_material_to_room_elements(target_id, mat_data)
            else:
                # Apply to specific object
                obj = self.objects_map.get(target_id)
                if obj:
                    mat = self._create_material(f"Mat_{target_id}", mat_data)
                    if obj.data.materials:
                        obj.data.materials[0] = mat
                    else:
                        obj.data.materials.append(mat)
                    count += 1

        print(f"✓ Materials applied: {count}")
        return count

    def _create_material(self, name: str, mat_data: Dict[str, Any]) -> bpy.types.Material:
        """
        Create PBR material

        Args:
            name: Material name
            mat_data: Material specification

        Returns:
            Created material
        """
        mat = bpy.data.materials.new(name=name)
        mat.use_nodes = True

        # Get principled BSDF
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if not bsdf:
            bsdf = mat.node_tree.nodes.new('ShaderNodeBsdfPrincipled')

        # Set properties
        base_color = mat_data.get("base_color", [0.8, 0.8, 0.8])
        roughness = mat_data.get("roughness", 0.5)
        metallic = mat_data.get("metallic", 0.0)

        bsdf.inputs['Base Color'].default_value = (*base_color, 1.0)
        bsdf.inputs['Roughness'].default_value = roughness
        bsdf.inputs['Metallic'].default_value = metallic

        return mat

    def _apply_material_to_room_elements(self, target: str, mat_data: Dict[str, Any]):
        """
        Apply material to room elements (floor, walls, ceiling)

        Args:
            target: Target element name
            mat_data: Material specification
        """
        mat = self._create_material(f"Mat_{target}", mat_data)

        # Find all relevant objects in collection
        for obj in self.collection.objects:
            if target in obj.name.lower():
                if obj.data and hasattr(obj.data, 'materials'):
                    if obj.data.materials:
                        obj.data.materials[0] = mat
                    else:
                        obj.data.materials.append(mat)

    def _get_or_create_collection(self, name: str) -> bpy.types.Collection:
        """Get or create a collection"""
        collection = bpy.data.collections.get(name)
        if not collection:
            collection = bpy.data.collections.new(name)
            bpy.context.scene.collection.children.link(collection)
        return collection

    def _clear_collection(self, collection: bpy.types.Collection):
        """Clear all objects from collection"""
        for obj in list(collection.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
