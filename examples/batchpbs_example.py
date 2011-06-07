from pbs import job
from time import sleep

job = job()
job.set_cluster(True)
job.set_nservers(2)
job.init()
job.submit_example()
job.reset()
job.submit_example()
print "DONE"
