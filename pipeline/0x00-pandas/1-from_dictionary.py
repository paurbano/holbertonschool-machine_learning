#!/usr/bin/env python3
'''created a pd.DataFrame from a dictionary'''
import pandas as pd

index = ["A", "B", "C", "D"]
d = {
    "First ": pd.Series([0.0, 0.5, 1.0, 1.5], index=index),
    "Second": pd.Series(['one', 'two', 'three', 'four'], index=index),
    }
df = pd.DataFrame(d)
