import numpy
#from bbspec.spec2d.arcmodel2D import arcmodel2D
from subprocess import call
from time import sleep
from sys import stdout
try: import pp
except ImportError,e :
    print "Parallel Python > Unavailable due to Exception: %s" % (e.args[0])
    print "                > Requires python package pp to be installed"

class job:
    
    ready = False
    defaultport = 60000
    
    def __init__(self):
        self.nservers = 1           #number of compute nodes
        self.port = 60000           #job listen port
        self.timeout = 60           #seconds to wait before freeing the node, after completion
        self.sleep = 20             #seconds to wait after qsub command before checking for active nodes
        self.password = '4-bigboss-05312011'  #job password (on compute nodes)
        self.active_nodes = None    #number of nodes actually available for job
        self.active_ncpus = None    #number of cpus actually available for job
        self.worker = None          #list of workers actually submitted to job server.
        self.result = None          #list of results received back from job server.
        self.function = None        #function to be sent to nodes for computation
        self.parameters = ()        #parameters to be sent with function to nodes for computation
        self.modules = ()           #external modules to be sent with function to nodes for computation
        self.cluster = False        #run on cluster (with or without localhost)
        self.localhost = True       #run workers on localhost (set to false for cluster computations)
        
    def set_nservers(self,nservers): self.nservers = nservers 
    def set_port(self,port): self.port = port 
    def set_password(self,password): self.password = password
    def set_cluster(self,cluster): 
        self.cluster = cluster
        self.localhost = not cluster
        if cluster: self.set_autodiscovery()

    def set_autodiscovery(self): 
        if self.port == job.defaultport: self.ppservers = ('*',)
        else: self.ppservers = ("*:%i" % self.port,)
    
    def init(self):
        if self.cluster: 
            if self.localhost: 
                self.server = pp.Server(ppservers=self.ppservers,secret=self.password)
                self.ncpus = self.server.get_ncpus()    #count number of cpus on localhost
            else:
                self.ncpus = 0                          #negate cpus on localhost
                self.server = pp.Server(ppservers=self.ppservers,ncpus=self.ncpus,secret=self.password)
        elif self.localhost:
            self.nservers = 1
            self.server = pp.Server()
            self.ncpus = self.server.get_ncpus()        #count number of actual cpus on localhost
        self.set_servers()
        self.reset()
        
    def reset(self):
        self.worker = []
        self.result = []
        self.active_nodes = self.server.get_active_nodes()
        self.active_ncpus =  sum([ncpu for ncpu in self.active_nodes.values()])
        cols = {'nodes':self.active_nodes,'ncpus':self.active_ncpus}
        print "JOB: active nodes = %(nodes)r => ncpus = %(ncpus)i" % cols
        job.ready = (self.active_ncpus > 0)

    def set_servers(self):
        if self.cluster:
            if self.localhost: nservers = self.nservers-1
            else: nservers = self.nservers
            for i in range(nservers): call(["/usr/bin/qsub","server.qsub"])
            if nservers>0:
                if self.localhost: w = 'with'
                else: w='without'
                stdout.write("JOB: Waiting for PBS on %i computational nodes, %s localhost [%i secs]" % (nservers,w,self.sleep))
                stdout.flush()
                for s in range(self.sleep):
                    sleep(1)
                    stdout.write(".")
                    stdout.flush()
                print "."
            else:
                if self.localhost: print "JOB: starting on localhost"
                else: print "JOB: nservers=0"
                        
    def set_function(self,function): 
        if not job.ready: self.init()
        else: self.reset()
        self.function = function

    def set_parameters(self,parameters=()): self.parameters = parameters 

    def set_modules(self,modules=()): self.modules = modules 

    def submit_parameters(self,parameters):
        self.set_parameters(parameters)
        self.submit()

    def submit_function_parameters(self,function,parameters):
        self.set_function(function)
        self.set_parameters(parameters)
        self.submit()
        
    def submit(self):
        if job.ready:
            worker = self.server.submit(self.function,self.parameters,modules=self.modules)
            self.worker.append(worker)

    def submit_example(self):
        if job.ready:
            for i in xrange(self.active_ncpus):
                eg = example()
                worker = self.server.submit(eg.sumStuff, (i*10, (i+1)*10, 0.5), modules=eg.modules)
                self.worker.append(worker)
            for worker in self.worker: self.result.append(worker())
            self.server.print_stats()
            print self.result
        else: 'ERROR: job server is not ready'

    def set_result(self):
        self.result = []
        for worker in self.worker: self.result.append(worker())

    def get_result(self):
        self.set_result()
        return self.result

class example:
    modules = ('numpy',)
    def sumStuff(self, start, end, step=1):
        return numpy.sum(numpy.arange(start, end, step))
