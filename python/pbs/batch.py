from os import system,environ

class batch:
    
    ready = True
    
    def __init__(self,arcid=None,flatid=None,camera=['r1','r2','b1','b2']):
        self.bundles_per_cam = 25
        self.set_arc_flat(arcid,flatid)
        self.set_camera(camera)
        self.submit()

    def set_arc_flat(self,arcid,flatid):
        self.arcid = arcid
        self.flatid = flatid
        
    def set_camera(self,camera):
        if type(camera) is list: self.camera=camera
        elif type(camera) is str: self.camera = [camera]
        else: self.camera = None
        if self.camera !=None: self.n_bund = len(self.camera)
        else: self.n_bund = 0
        
        
    def set_env(self,cam):
        if self.arcid!=None and self.flatid!=None:
            environ['arcid'] = cam + '-' + self.arcid
            environ['flatid'] = cam + '-' + self.flatid
        else: ready = False
        
    def submit(self):
        if self.camera != None:
            for cam in self.camera:
                self.set_env(cam)
                if self.ready:
                    for i_bund in range(self.bundles_per_cam):
                        cols = {'batchpbs_gh_psf':"batchpbs_gh_psf.py",'i_bund':i_bund}
                        command = "/usr/bin/qsub %(batchpbs_gh_psf)s -N psf-%(i_bund)i -v BUNDLE=%(i_bund)i -V -d `pwd`" % cols
                        system(command)
                        print command
