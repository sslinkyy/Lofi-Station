# Blender Scene Builder - Testing Guide

## 🎯 What's Ready to Test

You can now test the **Blender Scene Builder** without needing the AI worker running!

### What Works

- ✅ **Scene Builder** - Builds complete 3D scenes from JSON
- ✅ **Camera Creation** - Sets FOV, position, rotation
- ✅ **Room Geometry** - Floor, walls, ceiling planes
- ✅ **Object Placement** - Boxes, cards, lowpoly proxies
- ✅ **Lighting** - Key light, fill light, ambient
- ✅ **Materials** - PBR materials with color, roughness, metallic
- ✅ **Mock Data** - 2 realistic test scenes included

---

## 📋 Testing on Your Local PC (No GPU Needed!)

### Step 1: Install Blender Add-on

1. **Open Blender 4.0+**

2. **Install Add-on:**
   - Edit → Preferences → Add-ons
   - Click "Install..."
   - Browse to: `c:\Station\image-to-scene\blender_addon`
   - Select the entire `blender_addon` folder
   - Click "Install Add-on"

3. **Enable Add-on:**
   - Search for "Image to Scene"
   - Check the checkbox to enable it

4. **Verify:**
   - Press `N` in the 3D Viewport
   - You should see an "Image to Scene" tab

---

### Step 2: Test with Mock Data

1. **Open the Image to Scene panel** (N → Image to Scene)

2. **Click "Build from JSON (Test)"**

3. **Select a mock scene graph:**
   ```
   c:\Station\image-to-scene\tests\fixtures\mock_scene_graph_lofi_bedroom.json
   ```
   OR
   ```
   c:\Station\image-to-scene\tests\fixtures\mock_scene_graph_simple_room.json
   ```

4. **Watch it build!**
   - You'll see a new "Generated Scene" collection appear
   - Camera, room, objects, lights, materials all created

5. **Render the scene:**
   - Press `F12` (or Render → Render Image)
   - You should see a basic scene render

---

## 📸 What You Should See

### Lofi Bedroom Scene
**Contains:**
- Bed (box proxy with subdivision)
- Desk & Chair
- Plant (lowpoly cone)
- Lamp (lowpoly cylinder)
- Person (card cutout)
- Room (floor, walls, ceiling)
- 2 lights (key + fill)
- Window on back wall

**Expected result:** A cozy bedroom with warm lighting

### Simple Room Scene
**Contains:**
- Table (box proxy)
- Laptop (thin box)
- Room (floor, back wall)
- 2 lights (key + fill)

**Expected result:** Minimal scene with table and laptop

---

## 🎨 Customization

Once the scene is built, you can:

1. **Move objects** - Select and press `G`
2. **Rotate objects** - Select and press `R`
3. **Scale objects** - Select and press `S`
4. **Edit materials** - Shading workspace → Select object → Edit material nodes
5. **Adjust lighting** - Select lights → Modify strength/color
6. **Replace proxies** - Delete proxy, add real mesh, keep name

---

## 🔄 Re-running Tests

To test again with different JSON:

1. Select all objects in "Generated Scene" collection
2. Click "Build from JSON (Test)" again
3. Select same or different JSON file
4. It will clear and rebuild (if "Clear Existing" is checked)

---

## 🐛 Troubleshooting

### "No module named 'core'"
- Make sure you installed the entire `blender_addon` **folder**, not just `__init__.py`
- Restart Blender after installation

### "File not found"
- Use the full absolute path to the JSON file
- Ensure the files exist in `c:\Station\image-to-scene\tests\fixtures\`

### Objects appear in wrong positions
- This is expected with mock data (AI worker would provide real positions)
- You can manually adjust positions to test the scene builder

### No materials visible
- Switch to "Material Preview" or "Rendered" viewport shading
- Press `Z` → "Material Preview"

---

## 🧪 Testing Checklist

- [ ] Add-on installs without errors
- [ ] Panel appears in N-sidebar
- [ ] "Build from JSON" opens file browser
- [ ] Bedroom scene builds successfully
- [ ] Simple room scene builds successfully
- [ ] Camera is created and set as active
- [ ] Render produces an image
- [ ] Materials are visible in viewport
- [ ] Lights illuminate the scene
- [ ] Can manually adjust objects

---

## ✨ Next: Test AI Pipeline

Once your RTX 5080 PC is back, or if you want to test on Google Colab:

See: [tests/colab/test_ai_pipeline.ipynb](tests/colab/test_ai_pipeline.ipynb)

This will test the **full AI pipeline** (Depth + Segmentation + VLM) and generate **real scene graphs** from actual images!

---

## 📝 Reporting Issues

When testing, note:
- Which JSON file you used
- What error message appeared (if any)
- Screenshot of the built scene
- Blender version

This helps debug issues quickly!
