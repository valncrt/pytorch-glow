import math

def round_up_to_even(f):
    return math.ceil(f / 2.) * 2

print(round_up_to_even(104.55))