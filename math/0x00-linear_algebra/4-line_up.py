#!/usr/bin/env python3
''' '''


def add_arrays(arr1, arr2):
    '''sum two arrays '''
    array = []
    if len(arr1) != len(arr2):
        return None

    for i in range(len(arr1)):
        array.append(arr1[i] + arr2[i])

    return array
