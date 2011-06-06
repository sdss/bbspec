import numpy
#from bbspec.spec2d.arcmodel2D import arcmodel2D
from subprocess import call
try: import pp
except ImportError,e :
    print "Parallel Python > Unavailable due to Exception: %s" % (e.args[0])
    print "                > Requires python package pp to be installed"

class job:
    
    ready = False
    defaultport = 60000
    
    def __init__(self):
        self.nservers = 1           #number of compute nodes
        self.ppn = 8                #number of procs per compute node
        self.port = 60000           #job listen port
        self.timeout = 60           #seconds to wait before freeing the node, after completion
        self.password = '4-bigboss-05312011'  #job password (on compute nodes)
        self.ncpus = None           #number of cpus actually available for job
        self.worker = None          #list of workers actually submitted to job server.
        self.result = None          #list of results received back from job server.
        self.function = None        #function to be sent to nodes for computation
        self.parameters = ()        #parameters to be sent with function to nodes for computation
        self.modules = ()           #external modules to be sent with function to nodes for computation
        
    def set_nservers(self,nservers): self.nservers = nservers 
    def set_ppn(self,ppn): self.ppn = ppn 
    def set_port(self,port): self.port = port 
    def set_password(self,password): self.password = password
    
    def set_autodiscovery(self): 
        if self.port == job.defaultport: self.ppservers = ('*',)
        else: self.ppservers = ("*:%i" % self.port,)
    
    def init(self):
        self.set_autodiscovery()
        ncpus = self.nservers * self.ppn
        self.server = pp.Server(ppservers=self.ppservers,ncpus=ncpus,secret=self.password)
        self.set_servers()
        self.reset()
        
    def reset(self):
        self.worker = []
        self.result = []

    def set_servers(self):
        if self.nservers>1:
            for i in range(self.nservers): call(["/usr/bin/qsub","server.qsub"])
        self.ncpus = self.server.get_ncpus()
        print "job set servers ncpus = %i" % self.ncpus
        job.ready = (self.ncpus != None)
        
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
        self.init()
        if job.ready:
            for i in xrange(self.ncpus):
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