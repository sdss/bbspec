from pbs import job
from bbspec.spec2d.arcmodel2D import arcmodel2D
import os

job = job()
job.set_nservers(2)           #number of compute nodes requested    (default=1)
job.set_ppn(5)                #number of procs per compute node     (default=8)

if 'PBS_O_WORKDIR' in os.environ:
    os.chdir(os.environ['PBS_O_WORKDIR'])
else:
    print "$PBS_O_WORKDIR isn't set; staying where I am."

#- Hardcode arc and flat:
arcid  = 'r1-00126630'
flatid = 'r1-00126631'

m = arcmodel2D('00126630','00126630')
m.setarc_flat(arcid,flatid)
job.set_modules(m.modules)
job.set_function(m.model_arc)
if job.ready:
    try:
        for bundle in range(0,10):
            print 'SUBMIT JOB: model_arc for %s %s %d' % (arcid, flatid, bundle)
            job.submit_parameters((bundle, bundle))
        print job.get_result()
    except MemoryError,e: print e
    job.server.print_stats()
else: print 'ERROR: JOB SERVER OFFLINE' 

