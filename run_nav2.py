from omni.isaac.kit import SimulationApp

# Avvio headless
simulation_app = SimulationApp({"headless": True})

import omni.usd
from omni.isaac.core import World

USD_PATH = "/home/alessandro/Scrivania/saves/nav2.usd"

# Carica lo stage
omni.usd.get_context().open_stage(USD_PATH)

# Crea il world (equivale a premere Play)
world = World(stage_units_in_meters=1.0)
world.reset()

print("▶️ Simulazione avviata")

# Loop di simulazione
while simulation_app.is_running():
    world.step(render=False)

simulation_app.close()
