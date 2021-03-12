#!/usr/bin/env python3
'''Use of requests package to reuqest an API'''
import requests


def sentientPlanets():
    '''returns the list of names of the home planets of all sentient species
    '''
    r = requests.get('https://swapi-api.hbtn.io/api/species/')
    list_planets = []
    json = r.json()
    _next = json.get('next')
    # page = 1
    while _next:
        for result in json.get('results'):
            if result['designation'] == 'sentient' or\
               result['classification'] == 'sentient':
                planet = result['homeworld']
                if planet:
                    p = requests.get(planet)
                    pjson = p.json()
                    list_planets.append(pjson.get('name'))
        # print("page: {}: next:{}".format(page, _next))
        if _next == 1:
            break
        r = requests.get(_next)
        json = r.json()
        _next = json.get('next')
        if _next is None:
            _next = 1
        # print(_next)
        # page += 1
    # print(len(list_planets))
    return list_planets
