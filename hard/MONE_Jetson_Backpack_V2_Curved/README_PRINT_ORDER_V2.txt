MONE JETSON NANO BACKPACK - VERSION 2 (CURVED ROBOT BACK)

WHAT CHANGED
- Added 00_fit_test_curvature_strip.stl: low-cost strip to verify the robot-back curvature before printing the full adapter.
- Replaced the old flat 12 mm spacer with 04_curved_back_adapter_3_to_12mm.stl.
- The adapter is 3 mm thick at the center and gradually reaches 12 mm at the left/right edges. This creates a concave robot-contact surface for the Joy Robot's convex back.
- Added optional EVA cutting template. A 2-3 mm EVA/foam layer is recommended to absorb fit error and vibration.
- Backplate, outer cover, and fan grille geometry are unchanged from V1.

IMPORTANT ORIENTATION
- The adapter's FLAT face goes against 01_backplate_strap_mount.stl.
- The CURVED/CONCAVE face goes against the robot.
- Jetson remains vertical, with its connector edge facing downward.
- Do not place the battery in this backpack; keep heavy parts low in the robot.

PRINT ORDER
1. 00_fit_test_curvature_strip.stl (mandatory cheap test).
2. 01_backplate_strap_mount.stl (test Jetson and strap locations).
3. 04_curved_back_adapter_3_to_12mm.stl (only after strip fits).
4. Temporarily assemble adapter + backplate + real Jetson and test balance/cables.
5. 02_outer_cover.stl.
6. 03_fan_grille_40mm.stl.

TEST PRINT SETTINGS
00 curvature strip:
- PLA/PLA+, 0.28 mm layer, 3 walls, 10-15% infill, no support, 100% scale.
01 backplate first-fit:
- PLA/PLA+, 0.28 mm layer, 3 walls, 15% gyroid/grid, support from build plate only if required, 100% scale.

FINAL PRINT SETTINGS
01 backplate:
- PETG preferred or PLA+, 0.20 mm layer, 4 walls, 5 top/bottom, 30% gyroid.
04 curved adapter:
- PETG or PLA+, 0.20 mm layer, 4 walls, 25-30% gyroid. Print the FLAT face on the bed; supports normally not required.
02 outer cover:
- PLA+ preferred for appearance, 0.20 mm layer, 3-4 walls, 5 top/bottom, 15-20% gyroid. Print with the large open side on the bed; build-plate-only supports if slicer requires.
03 fan grille:
- PLA+/PETG, 0.20 mm layer, 3 walls, 20% infill, flat on bed.

ASSEMBLY
- Add 2-3 mm EVA foam between adapter and robot back.
- Use two 20-25 mm Velcro straps through the backplate slots.
- Tighten only enough to prevent movement; do not deform the robot shell.
- Ensure shoulder movement is free and bottom-facing Jetson connectors have cable clearance.
- Use four M3 x 12 screws for the cover.

DIMENSIONS
- Backpack cover: 150 W x 135 H x 72 D mm.
- Curved adapter: 125 W x 110 H mm; thickness 3 mm center to 12 mm edges.
- Curvature test strip: 125 W x 18 H mm; same 3-to-12 mm profile.
- Units: millimeters. Do not scale.
