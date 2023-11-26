import numpy as np

SIGMA = 0.05

class Trace():
    def __init__(self,workflow,id,time):
        self.workflow = workflow
        self.id = id
        self.tasks = []
        self.initial_task = None
        self.current_task = None
        self.poll_time = time

    def get_service(self,task_id):
        return self.workflow.app.task_graph.nodes[task_id]["subset"]

class TaskInstance():
    def __init__(self,task,trace,caller=None):
        self.task = task
        self.trace = trace
        self.progress = 0
        self.caller = caller
        self.active = True
        self.poll_time = 0
        self.load = []
        self.comm = []
        for s in self.task.subtasks:
            load = s[0] if s[2] == 0 else 0.0
            self.load.append(load)
            self.comm.append(s[2] == 0)

    # Executes task up to dt time points with the given cpu speed, starting from point 'start' (completion percentage of task)
    def execute_task(self,speed,start,dt):
        end = start+dt
        remaining_load,_ = self.next_load()
        estim,_ = self.estimation(speed,start)
        runall = True if estim == end else False
        t = start
        mem = []
        for i,s in zip(range(len(self.task.subtasks)),self.task.subtasks):
            if s[2] == 0 and self.load[i] > 0:
                if runall:
                    if self.load[i] == remaining_load:
                        progress = self.load[i]
                        duration = dt
                    else:
                        progress = min(self.load[i],dt*speed)
                        duration = min(self.load[i]/speed,dt)
                    remaining_load -= progress
                else:
                    progress = min(self.load[i],dt*speed)
                    duration = min(self.load[i]/speed,dt)
                self.load[i] -= progress
                dt -= duration
                mem += np.random.normal(s[1], SIGMA, int(t+duration)-int(t)).tolist()
                t += duration
            if dt == 0:
                break
        if mem == []:
            mem = (int(start+dt)-int(start)) * [0]
        return [mm if mm > 0 else 0 for mm in mem]

    # Load/operations left until next request/comm_task or the end of the task
    def next_load(self):
        load = 0
        for l,s,c in zip(self.load,self.task.subtasks,self.comm):
            # check if it's a comp or comm task
            if s[2] == 0:
                load += l
            elif s[2] == 1 and not c:
                return load, s[0]
        return load, None

    def estimation(self,speed,t):
        load,next = self.next_load()
        return t + float(load)/speed,next

    def isActive(self):
        return self.active

    def activate(self):
        self.active = True
        for c in range(len(self.comm)):
            if not self.comm[c]:
                self.comm[c] = True
                return

    def deactivate(self,t):
        self.active = False
        self.poll_time = t