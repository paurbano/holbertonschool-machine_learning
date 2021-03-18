#!/usr/bin/env python3
'''returns the list of school having a specific topic'''
import pymongo


def schools_by_topic(mongo_collection, topic):
    '''list of school having a specific topic
    '''
    list_schools = []
    mycol = mongo_collection.find({'topics': {'$all': [topics]}})
    for res in mycol:
        list_schools.append(res)
    return list_schools
