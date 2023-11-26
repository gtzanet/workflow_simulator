import random
import numpy as np
from simulator.runtime import *
import networkx as nx
import queue
from simulator.utils import *

MAXSUBTASKS = 10
COMM_PROB = 0.2
LOADMAX = 7000
LOADMIN = 1000
MEMMAX = 1
MAXLAMBDA = 0.2
MAXTASKTREES = 1
MAXSERVICES = 3

class Task():
    def __init__(self,id,depth):
        self.depth = depth
        self.id = id
        subtasknum = random.choice(range(MAXSUBTASKS))
        self.subtasks = [(0.1 * LOADMAX,np.random.random() * MEMMAX,0,)]
        prob = COMM_PROB
        for _ in range(depth):
            prob = prob * COMM_PROB
        for st in range(subtasknum):
            if np.random.random() > prob:
                self.subtasks.append((LOADMIN + np.random.random() * (LOADMAX-LOADMIN),np.random.random() * MEMMAX,0,))
            else:
                self.subtasks.append((-1,False,1,))
        self.subtasks.append((0.1 * LOADMAX,np.random.random() * MEMMAX,0,))

class Service():
    def __init__(self,id,node=None):
        self.node = node
        self.id = id
        self.threads = 1
        self.tasknum = 0
        self.reset()

    def deploy_task(self,task,t):
        if len(self.running_tasks) >= self.threads:
            #print("TASK ADDED TO QUEUE: "+str(task.task.id))
            self.queue.put(task)
            q = 1
        else:
            #print("STARTING TASK: "+str(task.task.id))
            self.running_tasks.append(task)
            if self not in self.node.active_services:
                self.node.active_services.append(self)
            q = 0
        self.queue_analytics.append((self.get_queue_size(),t))
        return q

    def get_speed(self):
        total_threads = sum([len(s.running_tasks) for s in self.node.active_services])
        power = min(float(self.node.cpu) / total_threads,1.0)
        return power * self.node.freq

    def get_queue_size(self):
        return self.queue.qsize()+len(self.running_tasks)

    def switch_task(self,t,fintask):
        self.running_tasks.remove(fintask)
        if not self.queue.empty():
            new_task = self.queue.get()
            self.running_tasks.append(new_task)
        else:
            if len(self.running_tasks) == 0:
                self.node.active_services.remove(self)
            new_task = None
        self.queue_analytics.append((self.queue.qsize()+len(self.running_tasks),t))
        return new_task

    def set_threads(self,threads):
        self.threads = threads
        #while len(self.running_tasks) > self.threads:
        #    task = random.choice(self.running_tasks)
        #    self.queue.queue.insert(0, task)
        #    self.running_tasks.remove(task)

    def reset(self):
        self.tasks = []
        self.running_tasks = []
        self.queue = queue.Queue()
        self.queue_analytics = [(0,0)]
        if self.node is not None:
            self.node.reset()

class Application():
    def __init__(self,servicenum=None,flownum=None):
        self.id = id
        self.services = []
        self.tasks = []
        self.task_graph = nx.DiGraph()
        self.workflows = []

        # Create services
        if servicenum is None:
            servicenum = random.choice(range(1,MAXSERVICES))
        print("Services: "+str(servicenum))
        for s in range(servicenum):
            self.services.append(Service(s))

        # Generate tasktrees and task graph
        if flownum is None:
            flownum = random.choice(range(1,MAXTASKTREES))
        print("Tasktrees: "+str(flownum))
        for tree in range(flownum):
            self.workflows.append(Workflow(self,tree,self.task_graph,self.tasks,servicenum))
            print(self.workflows[-1].nodes)

        # Create service graph
        self.service_graph = self.task_graph.copy()
        for n in self.service_graph.nodes:
            for m in self.service_graph.nodes:
                try:
                    if n != m and b.nodes[n]["subset"] == b.nodes[m]["subset"]:
                        b = nx.contracted_nodes(b, n, m)
                except Exception as e:
                    continue
        labels = {}
        for n in self.service_graph.nodes:
            labels[n] = self.service_graph.nodes[n]["subset"]
        print("Labels: "+str(labels))
        self.service_graph = nx.relabel_nodes(self.service_graph, labels)

    def deploy_services(self,map):
        for s in self.services:
            s.node = map[s.id]

    def reset(self,history=None):
        for s in self.services:
            s.reset()
        for f in self.workflows:
            log = history.pop(0) if history is not None else None
            f.reset(log)

    def get_history(self):
        return [f.trace_log for f in self.workflows]

class Workflow():
    def __init__(self,app,id,taskG,tasklist,servicenum,tree=None):
        self.app = app
        self.lam = (random.random()+1) / 2 * MAXLAMBDA
        self.id = id
        self.initial_task = None
        self.trace_log = []
        self.nodes = []
        self.reset()

        if tree is None:
            root = len(tasklist)
            #print("ROOT: "+str(root))
            self.nodes.append(root)
            taskG.add_node(root)
            taskG.nodes[root]["subset"] = random.choice(range(servicenum))
            self.initial_task = self.create_task(root,taskG,tasklist,taskG.nodes[root]["subset"],servicenum,0)
        else:
            self.tree = tree

    def create_task(self,node,taskG,tasklist,service,servicenum,depth):
        new_task = Task(node,depth)
        tasklist.append(new_task)
        for i,s in zip(range(len(new_task.subtasks)),new_task.subtasks):
            if s[2] == 1: # Comm subtask
                other_services = list(range(servicenum))
                other_services.remove(service)
                next_task = len(tasklist)
                self.nodes.append(next_task)
                taskG.add_edge(node,next_task)
                taskG.nodes[next_task]["subset"] = random.choice(other_services)
                new_task.subtasks[i] = (self.create_task(next_task,taskG,tasklist,taskG.nodes[next_task]["subset"],servicenum,depth+1),s[1],s[2],)
        return new_task

    def init_next_arrival(self,id,time):
        self.last_arrival = self.next_arrival
        if self.trace_history is None:
            self.next_arrival = self.last_arrival - np.log(np.random.random()) / self.lam
        else:
            self.next_arrival = self.trace_history.pop(0)
        self.trace_log.append(self.next_arrival)
        return Trace(self,id,time)

    def reset(self,trace_history=None):
        self.trace_history = trace_history
        self.last_arrival = 0
        if self.trace_history is None:
            self.next_arrival = - np.log(np.random.random()) / self.lam
        else:
            self.next_arrival = self.trace_history.pop(0)
        self.trace_log = [self.next_arrival]