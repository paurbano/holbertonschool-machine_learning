#!/usr/bin/env python3
"""
Test file
"""
availableShips = __import__('0-passengers').availableShips
ships = availableShips(3000)
for ship in ships:
    print(ship)
