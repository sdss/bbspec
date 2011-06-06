from pbs import job

job = job()
job.set_nservers(1)           #number of compute nodes requested    (default=1)
job.set_ppn(2)                #number of procs per compute node     (default=8)
job.submit_example()
if job.ready: print "Job Server is Ready -- # of worker cpus=%i" % job.ncpus
print "DONE"