# CAEAD - PyAEDT HFSS Integration
# Dr. B. Pritam Singh
# AI se design lo — HFSS mein seedha banana

import pyaedt
from pyaedt import Hfss
import numpy as np

print("CAEAD - PyAEDT HFSS Integration")
print("Dr. B. Pritam Singh")
print("=" * 50)

# ── Antenna parameters ────────────────────────────
# Yeh values AI (Inverse Design) se aayengi
# Abhi WiFi 2.4 GHz design use kar rahe hain

patch_length  = 20.85   # mm — AI predicted
patch_width   = 23.76   # mm — AI predicted
substrate_er  = 9.09    # — AI predicted
substrate_h   = 1.66    # mm — AI predicted
ground_length = 40.0    # mm
ground_width  = 40.0    # mm

print(f"\nDesign Parameters (from CAEAD AI):")
print(f"  Patch Length : {patch_length} mm")
print(f"  Patch Width  : {patch_width} mm")
print(f"  Substrate Er : {substrate_er}")
print(f"  Substrate H  : {substrate_h} mm")
print(f"  Target Freq  : 2.4 GHz (WiFi)")

print("\nLaunching HFSS...")

# ── Launch HFSS ───────────────────────────────────
hfss = Hfss(
    project_name="CAEAD_WiFi_2p4GHz",
    design_name="Patch_Antenna_AI",
    solution_type="DrivenModal",
    new_desktop_session=True,
    non_graphical=False   # HFSS window kholega
)

print("HFSS launched!")

# ── Variables ─────────────────────────────────────
hfss["patch_l"] = f"{patch_length}mm"
hfss["patch_w"] = f"{patch_width}mm"
hfss["sub_h"]   = f"{substrate_h}mm"
hfss["gnd_l"]   = f"{ground_length}mm"
hfss["gnd_w"]   = f"{ground_width}mm"

# ── Substrate ─────────────────────────────────────
print("Building substrate...")
substrate = hfss.modeler.create_box(
    origin      = ["-gnd_l/2", "-gnd_w/2", "0"],
    sizes       = ["gnd_l", "gnd_w", "sub_h"],
    name        = "Substrate",
    material    = "Rogers RO3010 (ANSYS)"
)
substrate.material_name = "Rogers RO3010 (ANSYS)"

# Custom Er if needed
hfss.materials.add_material("CAEAD_Substrate")
hfss.materials["CAEAD_Substrate"].permittivity = substrate_er
hfss.materials["CAEAD_Substrate"].dielectric_loss_tangent = 0.0023
substrate.material_name = "CAEAD_Substrate"

# ── Ground plane ──────────────────────────────────
print("Building ground plane...")
gnd = hfss.modeler.create_rectangle(
    orientation = "XY",
    origin      = ["-gnd_l/2", "-gnd_w/2", "0"],
    sizes       = ["gnd_l", "gnd_w"],
    name        = "Ground"
)
hfss.assign_perfect_e(gnd.name)

# ── Patch ─────────────────────────────────────────
print("Building patch...")
patch = hfss.modeler.create_rectangle(
    orientation = "XY",
    origin      = ["-patch_l/2", "-patch_w/2", "sub_h"],
    sizes       = ["patch_l", "patch_w"],
    name        = "Patch"
)
hfss.assign_perfect_e(patch.name)

# ── Feed port ─────────────────────────────────────
print("Creating feed port...")
feed_x = -patch_length/2 * 0.3   # 30% from edge

port = hfss.modeler.create_rectangle(
    orientation = "YZ",
    origin      = [f"{feed_x}mm", "-0.5mm", "0"],
    sizes       = ["1mm", "sub_h"],
    name        = "Feed_Port"
)

hfss.create_lumped_port_to_sheet(
    sheet_name      = "Feed_Port",
    portname        = "P1",
    reference_object_list = ["Ground"]
)

# ── Airbox ────────────────────────────────────────
print("Creating radiation boundary...")
airbox = hfss.modeler.create_box(
    origin = ["-gnd_l/2-10mm", "-gnd_w/2-10mm", f"-10mm"],
    sizes  = ["gnd_l+20mm", "gnd_w+20mm", f"{substrate_h+30}mm"],
    name   = "Airbox"
)
hfss.assign_radiation_boundary_to_objects("Airbox")

# ── Analysis setup ────────────────────────────────
print("Setting up analysis...")
setup = hfss.create_setup("CAEAD_Setup")
setup.props["Frequency"] = "2.4GHz"
setup.props["MaximumPasses"] = 10
setup.props["MaxDeltaS"] = 0.02

sweep = setup.add_sweep("CAEAD_Sweep")
sweep.props["RangeStart"] = "1GHz"
sweep.props["RangeEnd"]   = "4GHz"
sweep.props["RangeStep"]  = "10MHz"

# ── Save ──────────────────────────────────────────
print("\nSaving HFSS project...")
hfss.save_project()

print("\n" + "=" * 50)
print("CAEAD → HFSS Integration Complete!")
print("=" * 50)
print(f"Project : CAEAD_WiFi_2p4GHz.aedt")
print(f"Design  : Patch_Antenna_AI")
print(f"Target  : 2.4 GHz WiFi")
print(f"Patch   : {patch_length}mm x {patch_width}mm")
print(f"Er      : {substrate_er}")
print("=" * 50)
print("\nHFSS mein:")
print("1. Simulate → Analyze All")
print("2. Results → S11 plot dekho")
print("3. 2.4 GHz par resonance confirm karo")

# Keep HFSS open
input("\nEnter dabaо HFSS band karne ke liye...")
hfss.release_desktop()