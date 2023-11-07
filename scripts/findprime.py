#!/usr/bin/env python3

import sys
import math

def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

if __name__ == '__main__':
    num = int(sys.argv[1])
    while True:
        if is_prime(num):
            print(num)
            break
        num += 1

