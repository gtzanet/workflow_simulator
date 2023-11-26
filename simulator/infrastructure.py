import logging

class VirtualMachine():
    def __init__(self,id,cpu,ram,freq,cluster=None):
        self.id = id
        self.cpu = cpu
        self.ram = ram
        self.freq = freq
        self.cluster = cluster
        self.reset()

    def set_node(self,node):
        self.node = node

    def reset(self):
        self.active_services = []
        self.mem_metric = [0.0]
        self.cpu_metric = [0.0]

class Node():
    def __init__(self,cpu,mem,freq):
        self.cpu = cpu
        self.mem = mem
        self.freq = freq
        self.containers = []
        self.cpu_metrics = []
        self.mem_metrics = []

    def free_resources(self):
        used = self.used_resources()
        return (self.cpu-used[0],self.mem-used[1])

    def used_resources(self):
        cpu_used = 0
        mem_used = 0
        for c in self.containers:
            cpu_used += c.cpu
            mem_used += c.ram
        return cpu_used,mem_used

    def deploy_container(self,c):
        if c not in self.containers:
            free = self.free_resources()
            if free[0] >= c.cpu and free[1] >= c.ram:
                self.containers.append(c)
                c.set_node(self)
                return 0
            else:
                logging.info('Not enough resources for deploying the container')
                return 1
        else:
            logging.warning('Container already deployed')

    def inc_resources(self,container,cpu=None,mem=None):
        if container in self.containers:
            cpu = container.cpu if cpu is None else cpu
            mem = container.ram if mem is None else mem
            free = self.free_resources()
            if free[0]+container.cpu >= cpu and free[1]+container.ram >= mem:
                container.cpu = cpu
                container.ram = mem
            else:
                container.cpu += free[0]
                container.ram += free[1]
        else:
            logging.warning('Container not deployed in this node')

    def remove_containers(self):
        for c in self.containers:
            c.node = None
        self.containers = []

    def VCG_allocate(self,containers,bids):
        total_amount = self.cpu
        amounts = [c.cpu for c in containers]
        allocator = VCG_allocator()
        allocation,payments = allocator.allocate()
        self.remove_containers()
        for i,c in enumerate(containers):
            self.containers.append(c)
            c.node = self
            c.cpu = allocation[i]