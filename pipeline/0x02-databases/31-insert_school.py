#!/usr/bin/env python3
'''insert document'''
import pymongo


def insert_school(mongo_collection, **kwargs):
    '''inserts a new document in a collection
    '''
    _id = mongo_collection.insert_one(kwargs).insert_id
    return (_id)
