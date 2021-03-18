#!/usr/bin/env python3
'''changes all topics of a school documen'''
import pymongo


def update_topics(mongo_collection, name, topics):
    '''def update_topics(mongo_collection, name, topics)
    args:
        mongo_collection will be the pymongo collection object
        name (string) will be the school name to update
        topics (list of strings) will be the list of topics approached
            in the school
    '''
    mongo_collection.update({'name': name},
                            {'$set': {'topics': topics}},
                            multi=true)
