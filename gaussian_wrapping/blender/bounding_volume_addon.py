bl_info = {
    "name": "GW Bounding Volume Exporter",
    "author": "GaussianWrapping",
    "version": (1, 0),
    "blender": (3, 0, 0),
    "location": "View3D > N-Panel > GW Bounds",
    "description": "Export a mesh as a convex-hull bounding volume for primal_adaptive_meshing_extraction.py",
    "category": "3D View",
}

import bpy
import json
import os
from bpy.props import StringProperty, PointerProperty
from bpy.types import Panel, Operator, PropertyGroup


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

class GWBoundsProps(PropertyGroup):
    bounding_object: PointerProperty(
        name="Bounding Mesh",
        description="Mesh object that defines the bounding volume",
        type=bpy.types.Object,
        poll=lambda self, obj: obj.type == "MESH",
    )
    export_path: StringProperty(
        name="Export Path",
        description="Path to save the bounding volume JSON",
        default="//bounding_volume.json",
        subtype="FILE_PATH",
    )


# ---------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------

class GW_OT_ExportBoundingVolume(Operator):
    bl_idname = "gw.export_bounding_volume"
    bl_label = "Export Bounding Volume"
    bl_description = (
        "Export the selected mesh as a convex-hull bounding volume JSON "
        "for use with primal_adaptive_meshing_extraction.py --bounding_box_method blender"
    )

    def execute(self, context):
        props = context.scene.gw_bounds_props
        obj = props.bounding_object

        if obj is None:
            self.report({"ERROR"}, "No bounding mesh selected.")
            return {"CANCELLED"}

        if obj.type != "MESH":
            self.report({"ERROR"}, f"Object '{obj.name}' is not a mesh.")
            return {"CANCELLED"}

        export_path = bpy.path.abspath(props.export_path)
        export_dir = os.path.dirname(export_path)
        if export_dir and not os.path.exists(export_dir):
            self.report({"ERROR"}, f"Export directory does not exist: {export_dir}")
            return {"CANCELLED"}

        # Collect world-space vertices
        matrix = obj.matrix_world
        mesh = obj.data
        world_verts = [(matrix @ v.co).to_tuple() for v in mesh.vertices]

        if len(world_verts) < 4:
            self.report({"ERROR"}, "Bounding mesh must have at least 4 vertices to form a 3D convex hull.")
            return {"CANCELLED"}

        # Try to compute convex hull with scipy (available in most Blender 3.x+ Python envs)
        hull_simplices = None
        try:
            import numpy as np
            from scipy.spatial import ConvexHull
            arr = np.array(world_verts, dtype=np.float64)
            hull = ConvexHull(arr)
            hull_simplices = hull.simplices.tolist()
            # Reduce vertices to only hull vertices to keep the JSON small
            hull_vertex_indices = sorted(set(hull.vertices.tolist()))
            world_verts = arr[hull_vertex_indices].tolist()
            # Remap simplex indices
            idx_map = {old: new for new, old in enumerate(hull_vertex_indices)}
            hull_simplices = [[idx_map[i] for i in tri] for tri in hull_simplices]
        except ImportError:
            self.report(
                {"WARNING"},
                "scipy not found in Blender's Python — exporting raw mesh vertices. "
                "The extraction script will compute the convex hull from these.",
            )

        data = {
            "class_name": "GaussianWrappingBoundingVolume",
            "version": 1,
            "blender_object_name": obj.name,
            "vertices": world_verts,
            "hull_simplices": hull_simplices,  # None if scipy unavailable
        }

        with open(export_path, "w") as f:
            json.dump(data, f, indent=2)

        n_verts = len(world_verts)
        n_tris = len(hull_simplices) if hull_simplices is not None else "N/A (no scipy)"
        self.report(
            {"INFO"},
            f"Exported bounding volume: {n_verts} vertices, {n_tris} hull triangles → {export_path}",
        )
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------

class GW_PT_BoundsPanel(Panel):
    bl_label = "GW Bounds"
    bl_idname = "GW_PT_bounds_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "GW Bounds"

    def draw(self, context):
        layout = self.layout
        props = context.scene.gw_bounds_props

        layout.label(text="Bounding Volume Exporter", icon="MESH_CUBE")
        layout.separator()

        col = layout.column(align=True)
        col.label(text="Step 1: Import your input mesh as a reference.")
        col.label(text="Step 2: Create/sculpt a mesh that encloses the ROI.")
        col.label(text="Step 3: Select it below and export.")
        layout.separator()

        layout.prop(props, "bounding_object")
        layout.prop(props, "export_path")
        layout.separator()

        row = layout.row()
        row.scale_y = 1.5
        row.operator("gw.export_bounding_volume", icon="EXPORT")

        layout.separator()
        layout.label(text="Usage in extraction script:", icon="INFO")
        layout.label(text="--bounding_box_method blender")
        layout.label(text="--bounding_box_file <exported .json>")


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

classes = (
    GWBoundsProps,
    GW_OT_ExportBoundingVolume,
    GW_PT_BoundsPanel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.gw_bounds_props = PointerProperty(type=GWBoundsProps)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.gw_bounds_props


if __name__ == "__main__":
    register()
