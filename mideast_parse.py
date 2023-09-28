# Parses the Middle Eastern superfamily of languages. Such as English, Arabic, Hindi.
# Not designed to work on the Chinese or Mayan superfamilies.
import numba
import numpy as np

@numba.njit
def _fsm_core(x):
    N = x.size
    token_types = np.zeros(N, dtype=np.int32)
    inclusive_paren_nests = np.zeros(N, dtype=np.int32)
    in_quote = 0
    paren_lev = 0
    for i in range(N):
        ci = x[i]
        inclusive_paren_nests[i] = paren_lev
        if ci == 34 or ci == 39 or ci == 0x2018 or ci == 0x2019 or ci == 0x201C or ci == 0x201C: # Quotes
            token_types[i] = 3
            in_quote = 1 if in_quote==0 else 0
        elif in_quote:
            token_types[i] = 3
        elif ci == 10 or ci == 32 or ci == 13 or ci == 9: # Whitespace
            token_types[i] = 0
        elif ci >= 48 and ci <= 57 or (i<N-1 and (ci == 44 or ci == 46 or ci == 45) and x[i+1] >= 48 and x[i+1] <= 57): # numbers.
            token_types[i] = 3
        elif ci == 46 or ci == 58 or ci == 59 or ci == 33 or ci == 44 or ci == 63 or ci == 0x00BF: # Punctuation
            token_types[i] = 0 # Also 0.
        elif ci == 40 or ci == 91 or ci == 123: #([{
            token_types[i] = 4
            paren_lev = paren_lev + 1
            inclusive_paren_nests[i] = paren_lev
        elif ci == 41 or ci == 93 or ci == 125: #)]}
            paren_lev = paren_lev - 1
        else:
            token_types[i] = 1
    return token_types, inclusive_paren_nests

def fsm_parse(txt):
    x = np.frombuffer(txt.encode('UTF-32-LE'), dtype=np.uint32)
    return _fsm_core(x)
