#!/usr/bin/env python3
'''Use of requests package to reuqest an API'''
import requests
import sys
import time

if __name__ == '__main__':
    url = "https://api.spacexdata.com/v4/launches/upcoming"
    r = requests.get(url)
    launches = sorted(r.json(), key=lambda i: i['date_unix'])
    date_unix = launches[0]['date_unix']
    for i in r.json():
        if i['date_unix'] == date_unix:
            launch_name = i['name']
            date = i['date_local']
            rocket_id = i['rocket']
            launchpad_id = i['launchpad']
            break
    url = "https://api.spacexdata.com/v4/rockets/{}".format(rocket_id)
    r = requests.get(url)
    rocket_name = r.json()['name']
    url = "https://api.spacexdata.com/v4/launchpads/{}".format(launchpad_id)
    r = requests.get(url)
    lpad_name = r.json()['name']
    lpad_locality = r.json()['locality']
    print("{} ({}) {} - {} ({})".format(launch_name, date, rocket_name,
                                        lpad_name, lpad_locality))
