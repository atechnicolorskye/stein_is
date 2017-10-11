#!/usr/bin/python

import multiprocessing
import time

def square(x):
    # This is is reeeeally slow way to square numbers.
    time.sleep(0.5)
    return x**2

print "Creating pool with 3 workers"
pool = multiprocessing.Pool(processes=3)

print "Invoking apply(square, 3)"
print "Result: %s" % (pool.apply(square, [3]),)

print "Invoking apply_async(square, 4)"
result = pool.apply_async(square, [4])
print "Waiting for result..."
start_time = time.time()
print "Result: %s (%.2f secs)" % (result.get(), time.time() - start_time)

print "Invoking map(square, [5,6,7,8,9,10])"
print "Result: %r" % (result.get(),)
print "Invoking map_async(square, [11, 12, 13, 14, 15, 16])"
result = pool.map_async(square, [11, 12, 13, 14, 15, 16])
print "Waiting for result..."
start_time = time.time()
print "Result: %r (%.2f secs)" % (result.get(), time.time() - start_time)

print "Invoking apply_async(square, X) once each for X in 17-24"
results = [(i, pool.apply_async(square, [i])) for i in xrange(17, 25)]
start_time = time.time()
for i, result in results:
    print "Result (%d): %s (%.2f secs)" % (i, result.get(), time.time() - start_time)
