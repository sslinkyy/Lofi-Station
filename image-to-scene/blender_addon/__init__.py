"""
Image-to-Scene Blender Add-on
Transform images into 3D Blender scenes using AI
"""

bl_info = {
    "name": "Image to Scene",
    "author": "Image-to-Scene Team",
    "version": (0, 1, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > Image to Scene",
    "description": "AI-powered 3D scene reconstruction from images",
    "category": "3D View",
    "doc_url": "https://github.com/sslinkyy/Lofi-Station",
    "tracker_url": "https://github.com/sslinkyy/Lofi-Station/issues",
}

import bpy
from bpy.props import StringProperty, EnumProperty, IntProperty, BoolProperty, FloatProperty
from bpy.types import AddonPreferences

# Import modules
from .core import scene_builder
# from . import ui
# from . import utils


class ImageToScenePreferences(AddonPreferences):
    """Add-on preferences"""
    bl_idname = __name__

    worker_url: StringProperty(
        name="AI Worker URL",
        description="URL of the AI worker service",
        default="http://127.0.0.1:8000",
    )

    worker_mode: EnumProperty(
        name="Worker Mode",
        description="How to connect to AI worker",
        items=[
            ('LOCAL', "Local", "Run worker on this machine"),
            ('REMOTE', "Remote", "Connect to remote worker"),
            ('CLOUD', "Cloud", "Use cloud API (not implemented)"),
        ],
        default='LOCAL',
    )

    style_preset: EnumProperty(
        name="Default Style",
        description="Default rendering style",
        items=[
            ('LOFI', "Lofi Cozy", "Stylized, soft, YouTube lofi aesthetic"),
            ('REALISTIC', "Realistic", "Accurate materials and lighting"),
            ('ARCHITECTURAL', "Architectural", "Clean, precise, CAD-like"),
        ],
        default='LOFI',
    )

    processing_mode: EnumProperty(
        name="Processing Mode",
        description="Speed vs quality tradeoff",
        items=[
            ('FAST', "Fast Preview", "Quick results, lower quality"),
            ('BALANCED', "Balanced", "Good quality, reasonable speed"),
            ('HQ', "High Quality", "Best results, slower"),
        ],
        default='BALANCED',
    )

    max_verify_iterations: IntProperty(
        name="Max Verification Iterations",
        description="Maximum number of verification/refinement loops",
        default=5,
        min=0,
        max=20,
    )

    auto_cleanup: BoolProperty(
        name="Auto Cleanup",
        description="Automatically clean up temporary files",
        default=True,
    )

    cache_results: BoolProperty(
        name="Cache Results",
        description="Cache AI worker results for faster re-processing",
        default=True,
    )

    def draw(self, context):
        layout = self.layout

        layout.label(text="AI Worker Settings:")
        layout.prop(self, "worker_mode")
        layout.prop(self, "worker_url")

        layout.separator()
        layout.label(text="Default Settings:")
        layout.prop(self, "style_preset")
        layout.prop(self, "processing_mode")
        layout.prop(self, "max_verify_iterations")

        layout.separator()
        layout.label(text="Options:")
        layout.prop(self, "auto_cleanup")
        layout.prop(self, "cache_results")

        # Test connection button
        layout.separator()
        row = layout.row()
        row.operator("image_to_scene.test_connection", text="Test Worker Connection")


# Simple operator for testing connection
class IMAGE_TO_SCENE_OT_test_connection(bpy.types.Operator):
    """Test connection to AI worker"""
    bl_idname = "image_to_scene.test_connection"
    bl_label = "Test Connection"
    bl_description = "Test connection to AI worker service"

    def execute(self, context):
        import requests
        from . import utils

        prefs = context.preferences.addons[__name__].preferences

        try:
            response = requests.get(f"{prefs.worker_url}/health", timeout=5)

            if response.ok:
                data = response.json()
                self.report({'INFO'}, f"✓ Connected! Status: {data.get('status')}")
                return {'FINISHED'}
            else:
                self.report({'ERROR'}, f"✗ Connection failed: {response.status_code}")
                return {'CANCELLED'}

        except requests.exceptions.ConnectionError:
            self.report({'ERROR'}, f"✗ Cannot connect to {prefs.worker_url}")
            self.report({'ERROR'}, "Make sure AI worker is running: python ai_worker/main.py")
            return {'CANCELLED'}

        except Exception as e:
            self.report({'ERROR'}, f"✗ Error: {str(e)}")
            return {'CANCELLED'}


class IMAGE_TO_SCENE_OT_build_from_json(bpy.types.Operator):
    """Build scene from JSON file (for testing)"""
    bl_idname = "image_to_scene.build_from_json"
    bl_label = "Build Scene from JSON"
    bl_description = "Load a scene graph JSON file and build the 3D scene"
    bl_options = {'REGISTER', 'UNDO'}

    filepath: StringProperty(
        name="File Path",
        description="Path to scene graph JSON file",
        subtype='FILE_PATH',
    )

    filter_glob: StringProperty(
        default="*.json",
        options={'HIDDEN'},
    )

    clear_existing: BoolProperty(
        name="Clear Existing",
        description="Clear existing objects in the generated scene collection",
        default=True,
    )

    def execute(self, context):
        import json
        from pathlib import Path

        try:
            # Load JSON
            with open(self.filepath, 'r') as f:
                scene_graph = json.load(f)

            self.report({'INFO'}, f"Loaded scene graph from: {Path(self.filepath).name}")

            # Build scene
            builder = scene_builder.SceneBuilder(scene_graph, collection_name="Generated Scene")
            results = builder.build_scene(clear_existing=self.clear_existing)

            if results["success"]:
                self.report({'INFO'},
                    f"✓ Scene built: {results['objects_created']} objects, "
                    f"{results['lights_created']} lights, {results['materials_created']} materials")
                return {'FINISHED'}
            else:
                errors = ", ".join(results["errors"])
                self.report({'ERROR'}, f"✗ Scene build failed: {errors}")
                return {'CANCELLED'}

        except FileNotFoundError:
            self.report({'ERROR'}, f"✗ File not found: {self.filepath}")
            return {'CANCELLED'}

        except json.JSONDecodeError as e:
            self.report({'ERROR'}, f"✗ Invalid JSON: {str(e)}")
            return {'CANCELLED'}

        except Exception as e:
            self.report({'ERROR'}, f"✗ Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}


# Placeholder panel (will be expanded later)
class IMAGE_TO_SCENE_PT_main_panel(bpy.types.Panel):
    """Main Image to Scene panel"""
    bl_label = "Image to Scene"
    bl_idname = "IMAGE_TO_SCENE_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Image to Scene"

    def draw(self, context):
        layout = self.layout

        layout.label(text="🚀 Image to Scene (v0.1)", icon='SCENE_DATA')
        layout.separator()

        # Status
        prefs = context.preferences.addons[__name__].preferences
        box = layout.box()
        box.label(text="Worker Status:", icon='NETWORK_DRIVE')
        box.label(text=f"  {prefs.worker_url}")
        box.operator("image_to_scene.test_connection", text="Test Connection")

        layout.separator()

        # Scene building
        box = layout.box()
        box.label(text="Scene Builder:", icon='SCENE')
        box.operator("image_to_scene.build_from_json", text="Build from JSON (Test)", icon='IMPORT')

        layout.separator()

        # Placeholder for main functionality
        layout.label(text="Full Pipeline (Coming Soon):")
        layout.label(text="  • Upload Image")
        layout.label(text="  • Set Scale Measurement")
        layout.label(text="  • Analyze & Build Scene")
        layout.label(text="  • Manual Refinement")
        layout.label(text="  • Verification Loop")


# Registration
classes = (
    ImageToScenePreferences,
    IMAGE_TO_SCENE_OT_test_connection,
    IMAGE_TO_SCENE_OT_build_from_json,
    IMAGE_TO_SCENE_PT_main_panel,
)


def register():
    """Register add-on"""
    for cls in classes:
        bpy.utils.register_class(cls)

    print("Image-to-Scene add-on registered")


def unregister():
    """Unregister add-on"""
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

    print("Image-to-Scene add-on unregistered")


if __name__ == "__main__":
    register()
