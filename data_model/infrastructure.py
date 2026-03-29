class Pod():
    def __init__(self,node,cpu=500,mem=500,replicas=1):
        self.node = node
        self.cpu = cpu
        self.mem = mem
        self.replicas = replicas

class ComputeNode():
    def __init__(self,name,cpu,mem,storage,cluster=None):
        self.name = name
        self.cpu = cpu
        self.mem = mem
        self.storage = storage
        self.cluster = cluster

class Cluster():
    def __init__(self,name):
        self.name = name

class NetworkNode():
    def __init__(self,name,type):
        self.name = name
        self.type = type

class PhysicalLink():
    def __init__(self,node1,node2,bandwidth):
        self.node1 = node1
        self.node2 = node2
        self.bandwidth = bandwidth

class VirtualLink():
    def __init__(self,node1,node2,controller):
        self.node1 = node1
        self.node2 = node2
        self.controller = controller

class Controller():
    def __init__(self,name,type,network_node):
        self.name = name
        self.type = type
        self.network_node = network_node