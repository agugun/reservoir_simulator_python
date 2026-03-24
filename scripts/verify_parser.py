import os
import sys
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.io import EclipseParser
from src.core import Simulator

def verify():
    print("Testing EclipseParser...")
    data_file = 'data/sample/sample_model.DATA'
    
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found.")
        return
        
    parser = EclipseParser(data_file)
    model = parser.build_model()
    
    print(f"Model loaded successfully!")
    print(f"Grid: {model.grid.nx}x{model.grid.ny}x{model.grid.nz}")
    print(f"Wells: {len(model.wells)} wells found.")
    for well in model.wells:
        print(f" - Well: {well.name} at {well.location} with rate {well.rate}")
        
    print("\nInitializing Simulator...")
    sim = Simulator(model)
    
    print("Running 3 simulation steps...")
    for i in range(1, 4):
        p = sim.step(dt=10.0)
        avg_pressure = np.mean(p)
        print(f"Step {i} | Avg Pressure: {avg_pressure:.2f} psi")

if __name__ == "__main__":
    verify()
