#!usr/bin/env python3
'''lists all documents in a collection'''
import pymongo


def list_all(mongo_collection):
    '''lists all documents in a collection:
    '''
    return mongo_collection.find()
