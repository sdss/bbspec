#!/usr/bin/env python

"""
results = xmap(fn, inputs) run in parallel processes

Notes:
 - doesn't support keyword arguments 
"""

from time import time
import multiprocessing
from multiprocessing import Process, Queue, cpu_count

def xmap(fn, args, nprocs=None, verbose=False):
    """
    return results = map(fn, args) = [fn(x) for x in args] run in parallel
    """
    return MapServer(fn, args, nprocs=nprocs, verbose=verbose).get_results()

class MapServer(object):
    """
    Class for handling parallel processing of an xmap command
    """
    def __init__(self, fn, args, nprocs=None, verbose=False):
        self.fn = fn
        self.args = args
        self.njobs = 0
        self.nresults = 0
        self.jobs = list()
        self.results = [None,] * len(self.args)
        self.results_queue = Queue()
        self.verbose = verbose
        if nprocs is None:
            self.nprocs = cpu_count()
        else:
            self.nprocs = nprocs

        self.start_jobs()

    def start_jobs(self):
        #- Start first batch of jobs
        for i in range( min(self.nprocs, len(self.args)) ):
            self._start_job()

        #- Get results, submit remaining jobs
        while self.nresults < self.njobs:
            self._get_result()
            if len(self.args) > self.njobs:
                self._start_job()
                
        #- Cleanup jobs
        # for job in self.jobs:
        #     job.join()

    def get_results(self, full_output=False):
        """
        Return array results[i] = fn(args[i])
        
        If full_output == True, return list of dictionaries with keys
          - fn      : the function evaluated
          - args    : the input arguments used
          - results : return value of fn(args)
          - ijob    : job number (should be same as array index)
          - time    : wallclock time to run
        """
        if full_output:
            return self.results
        else:
            return [x['results'] for x in self.results]

    def _start_job(self):
        """
        Start the next parallel job, increment njob counter
        """
        xargs = (self.fn, self.results_queue, self.njobs) + tuple(self.args[self.njobs])
        job = Process(target=fn_wrapper, args=xargs)
        
        if self.verbose:
            print 'Starting job %d' % self.njobs
        job.start()
        self.jobs.append(job)
        self.njobs += 1
        return job

    def _get_result(self):
        """
        Get a single result from the results_queue and insert it
        into self.results
        """
        r = self.results_queue.get()
        ijob = r['ijob']
        if self.verbose:
            print 'Got result %d' % ijob
        self.results[ijob] = r
        self.jobs[ijob].join()  #- join job to finish it off
        self.nresults += 1
        return r

def fn_wrapper(fn, results_queue, ijob, *args):
    """
    Run fn(*args) and put results into results_queue

    Results are a dictionary keyed by:
      - args    : the input arguments
      - results : the return value of fn(*args)
      - ijob    : the input job number identifier
      - time    : the wallclock runtime of fn(*args)

    This is implemented separate from the MapServer class so that the
    entire MapServer object isn't passed to each parallel process.
    """
    t0 = time()
    try:
        r = fn(*args)
    except Exception, e:
        print e
        r = None
    tx = time() - t0

    results = dict(args=args, results=r, ijob=ijob, time=tx)
    results_queue.put(results)


if __name__ == '__main__':

    from random import random
    from time import sleep
    
    def test_fn(t, name):
        sleep(t)
        return name

    inputs = list()
    inputs.append((random(), 'andy') )
    inputs.append((random(), 'bob') )
    inputs.append((random(), 'charlie') )
    inputs.append((random(), 'daryl') )
    inputs.append((random(), 'edward') )
    inputs.append((random(), 'frank') )
    inputs.append((random(), 'gary') )
    inputs.append((random(), 'ian') )
    print xmap(test_fn, inputs, verbose=True)
