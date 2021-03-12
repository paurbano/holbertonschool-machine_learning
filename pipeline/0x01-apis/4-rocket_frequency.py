#!/usr/bin/env python3
'''Use of requests package to reuqest an API'''
import requests
import sys
import time


if __name__ == '__main__':
    url = "https://api.spacexdata.com/v4/launches"
    r = requests.get(url)
    launches = {}
    for i in r.json():
        if i['rocket'] not in launches:
            launches[i['rocket']] = 1
        else:
            launches[i['rocket']] += 1
    url = "https://api.spacexdata.com/v4/rockets/"
    r = requests.get(url)
    rockets = []
    for i in r.json():
        if i['id'] in launches:
            rockets.append({'rocket': i['name'],
                            'launches': launches[i['id']]})
        else:
            continue
    launches = sorted(rockets, key=lambda i: i['rocket'])
    launches = sorted(rockets, key=lambda i: i['launches'], reverse=True)
    for i in launches:
        print("{}: {}".format(i['rocket'], i['launches']))
