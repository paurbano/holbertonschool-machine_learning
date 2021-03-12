#!/usr/bin/env python3
'''returns the list of ships'''
import requests


def availableShips(passengerCount):
    '''returns the list of ships that can hold a given number of passengers
    Args:
        passengerCount: number of passengers
    '''
    r = requests.get('https://swapi-api.hbtn.io/api/starships/')
    list_ships = []
    json = r.json()
    _next = json.get('next')
    while _next:
        for result in json.get('results'):
            try:
                passengers = result['passengers'].replace(',','')
                if int(passengers) >= passengerCount:
                    list_ships.append(result['name'])
            except ValueError:
                pass
        r = requests.get(_next)
        json = r.json()
        _next = json.get('next')
    return list_ships
