#!/usr/bin/env python3
'''Use of requests package to reuqest an API'''
import requests
import time
import sys

if __name__ == "__main__":
    if len(sys.argv) > 1:
        url = sys.argv[1]

    r = requests.get(url)
    if r.status_code == 403:
        rate_limit = r.headers.get('X-Ratelimit-Reset')
        xmin = (int(rate_limit) - int(time.time())) / 60
        print("Reset in {} min".format(int(xmin)))
    elif r.status_code == 404:
        print("Not found")
    else:
        json = r.json()
        print("{}".format(json.get('location')))
