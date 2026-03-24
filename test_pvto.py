import os
from opm.io import Parser
deck = Parser().parse('data/sample/SPE1CASE1.DATA')

print("PVDG:")
if 'PVDG' in deck:
    for i, row in enumerate(deck['PVDG']):
        print(f"Row {i}: {row[0].get_raw_data_list()}")

print("\nPVTO:")
if 'PVTO' in deck:
    for i, row in enumerate(deck['PVTO']):
        print(f"Row {i}: Rs={row[0].get_raw(0)}")
        vals = row[1].get_raw_data_list()
        print(f"  Data length {len(vals)}: {vals}")
