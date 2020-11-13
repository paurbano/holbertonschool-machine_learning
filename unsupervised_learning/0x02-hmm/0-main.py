#!/usr/bin/env python3

import numpy as np
markov_chain = __import__('0-markov_chain').markov_chain

if __name__ == "__main__":
    P = np.array([[0.25, 0.2, 0.25, 0.3], [0.2, 0.3, 0.2, 0.3], [0.25, 0.25, 0.4, 0.1], [0.3, 0.3, 0.1, 0.3]])
    s = np.array([[1, 0, 0, 0]])
    # assert (np.array_equal(markov_chain(P, s, 300), [[0.2494929,0.26335362,0.23394185,0.25321163]])), ""
    print([[0.2494929,0.26335362,0.23394185,0.25321163]])
    print(markov_chain(P, s, 300))
