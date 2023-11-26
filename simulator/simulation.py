from matplotlib import pyplot as plt
import random
from simulator.utils import *
from simulator.runtime import *

class Event():
    def __init__(self,type,time,workflow=None,task=None,next_task=None):
        self.type = type
        self.workflow = workflow
        self.task = task
        self.next_task = next_task
        self.time = time

class Simulation():
    def __init__(self,apps,units,iterations,step_size,alloc_step_size,timeout,alloc_method=0,DEBUG=False):
        self.iterations = iterations
        self.step_size = step_size
        self.alloc_step_size = alloc_step_size
        self.timeout = timeout
        self.alloc_method = alloc_method
        self.DEBUG = DEBUG
        self.active_traces = []
        self.apps = apps
        self.CUs = units
        self.last_trace_id = 0
        self.history = {}

    def forward(self,event):
        t = event.time
        secs = int(t) - int(self.t)
        dt = t - self.t
        self.t = t
        if self.DEBUG:
            print("\n######################################################################\n")
            print("TIME: "+str(t))
        for s in self.apps[0].services:
            if len(s.node.mem_metric) != int(t)+1:
                s.node.mem_metric += secs*[0]
                #if DEBUG:
                #    print("Service: "+str(s.id))
                #    print("Active services: "+str([ss.id for ss in s.node.active_services]))
                #    print("Running tasks: "+str([len(ss.running_tasks) for ss in s.node.active_services]))
                total_threads = sum([len(ss.running_tasks) for ss in s.node.active_services])
                usage = min(total_threads,s.node.cpu)
                #if DEBUG:
                #    print("Threads: "+str(total_threads))
                #    print("Usage: "+str(usage))
                s.node.cpu_metric += secs*[usage]
            for tt in s.running_tasks:
                if self.DEBUG:
                    print("Service: "+str(s.id)+", Running tasks: "+str([rt.task.id for rt in s.running_tasks]))
                if tt.isActive():
                    mem_metric = tt.execute_task(s.get_speed(),t-dt,dt)
                    if len(mem_metric) != secs:
                        raise Exception("Memory metrics length is incorrect")
                    #if DEBUG:
                    #    print("Measurements got: "+str(len(mem_metric)))
                    #    print("Measurements on CU: "+str(len(s.node.mem_metric)))
                    #    print("dt = "+str(dt))
                    for x in range(secs):
                        s.node.mem_metric[int(t)-secs+x] += mem_metric[x]
                elif self.DEBUG:
                    print("Task "+str(tt.task.id)+" is not active")
        if event.type == 0:
            if self.DEBUG:
                print("TRACE ARRIVAL")
                print("New arrival at time="+str(t)+" from tree: "+str(event.workflow.nodes))
            new_trace = event.workflow.init_next_arrival(self.last_trace_id,t)
            if self.DEBUG:
                print("Trace ID = "+str(self.last_trace_id))
            task = new_trace.workflow.initial_task
            s = self.apps[0].services[new_trace.get_service(task.id)]
            instance = TaskInstance(task,new_trace)
            q = s.deploy_task(instance,self.t)
            new_trace.initial_task = instance
            new_trace.current_task = instance
            if q == 0:
                self.history[new_trace.id] = {"status": 0, new_trace.current_task.task.id: [t,t]}
            else: # processor was full, task added to queue
                self.history[new_trace.id] = {"status": 0, new_trace.current_task.task.id: [t]}
            self.last_trace_id += 1
            self.active_traces.append(new_trace)
        elif event.type == 1:
            if self.DEBUG:
                print("SUBTASK DEPARTURE")
            current_task = event.task
            next_task = event.next_task # next task as a child of the ongoing task (different service)
            s = self.apps[0].services[current_task.trace.get_service(current_task.task.id)]

            # The executed task terminates
            if next_task is None:
                queued_task = s.switch_task(self.t,current_task) # Current service: new task added to the service's running tasks from the queue
                if queued_task is not None:
                    self.history[queued_task.trace.id][queued_task.task.id].append(t)
                self.history[current_task.trace.id][current_task.task.id].append(t)
                if self.DEBUG:
                    print("TRACE: "+str(current_task.trace.id)+", TASK COMPLETED: "+str(current_task.task.id)+" at time="+str(t))
                if current_task.caller is not None:
                    current_task.caller.activate()
                    self.history[current_task.trace.id][current_task.caller.task.id].append(t)
                elif self.history[current_task.trace.id]["status"] == 0:
                    self.history[current_task.trace.id]["status"] = 1
                    self.active_traces.remove(current_task.trace)
            # The executed task calls another task (i.e. it is deactivated waiting for a response)
            else:
                if self.DEBUG:
                    print("CALLING TASK: "+str(next_task.id))
                next_instance = TaskInstance(next_task,current_task.trace,current_task)
                next_s = self.apps[0].services[current_task.trace.get_service(next_task.id)]
                q = next_s.deploy_task(next_instance,self.t)
                current_task.trace.current_task = next_instance # is this used??
                if q == 0:
                    self.history[current_task.trace.id][next_task.id] = [t,t]
                else: # processor was full, task added to queue
                    self.history[current_task.trace.id][next_task.id] = [t]
                current_task.deactivate(t)
                self.history[current_task.trace.id][current_task.task.id].append(t)
        elif event.type == 2:
            if self.DEBUG:
                print("SUBTASK TIMEOUT")
            current_task = event.task
            s = self.apps[0].services[get_service(self.apps[0],current_task.task.id)]
            queued_task = s.switch_task(self.t,current_task) # Current service: new task added to the service's running tasks from the queue
            if queued_task is not None:
                self.history[queued_task.trace.id][queued_task.task.id].append(t)
            #self.history[current_task.trace.id][current_task.task.id].append(t)
            if self.DEBUG:
                print("TRACE: "+str(current_task.trace.id)+", TASK TIMED OUT: "+str(current_task.task.id)+" at time="+str(t))
            self.active_traces.remove(current_task.trace)
            self.history[current_task.trace.id]["status"] = 2
        if self.DEBUG:
            print("History:")
            print(self.history)
            print("Node active services: "+str([ss.id for ss in s.node.active_services]))
            print("Active traces: "+str([x.id for x in self.active_traces]))

    def run(self,agents=None):
        self.t = 0
        it = 0
        event = 0
        prev_step_time = 0
        arrivals = len(self.apps[0].services)*[0]
        departures = len(self.apps[0].services)*[0]
        late_metrics = len(self.apps[0].services)*[0]
        sum_late_metrics = len(self.apps[0].services)*[0]
        thrput_metrics = len(self.apps[0].services)*[0]
        arrival_rate = len(self.apps[0].services)*[0]
        departure_rate = len(self.apps[0].services)*[0]
        avg_late_metrics = []
        avg_thrput_metrics = []
        avg_arrival_rate = []
        avg_departure_rate = []
        resources = []
        for s in self.apps[0].services:
            avg_late_metrics.append([])
            avg_thrput_metrics.append([])
            avg_arrival_rate.append([])
            avg_departure_rate.append([])
            resources.append([])
        total_traces = len(self.apps[0].workflows)*[0]
        trace_success_metrics = len(self.apps[0].workflows)*[0]
        trace_fail_metrics = len(self.apps[0].workflows)*[0]
        sum_trace_success = len(self.apps[0].workflows)*[0]
        avg_trace_success = []
        for tt in self.apps[0].workflows:
            avg_trace_success.append([])
        print()
        for it in range(1,self.iterations+1):
            event = Event(0,-1)
            if self.DEBUG:
                print("\n\n######################################################################\n")
                print("PREVIOUS TIME: "+str(self.t))
                #for tf in self.active_traces:
                #    print("Trace "+str(tf.id)+", current task: "+str(tf.current_task.task.id)+ " at service: "+str(tf.get_service(tf.current_task.task.id)))
            for app in self.apps:
                for tt in app.workflows:
                    if self.DEBUG:
                        print("New trace from workflow "+str(tt.nodes)+" at time: "+str(tt.next_arrival))
                    if event.time == -1 or tt.next_arrival < event.time:
                        event.type = 0
                        event.time = tt.next_arrival
                        event.workflow = tt
                        event.task = None
                        event.next_task = None
            for app in self.apps:
                for s in app.services:
                    for rt in s.running_tasks:
                        if rt.isActive():
                            speed = s.get_speed()
                            estim,next = rt.estimation(speed,self.t)
                            if self.DEBUG:
                                print("Task "+str(rt.task.id)+" will end or call another task in trace: "+str(rt.trace.id)+" --> "+str(estim))
                            if event.time == -1 or estim < event.time:
                                event.workflow = rt.trace.workflow
                                event.type = 1
                                event.time = estim
                                event.task = rt
                                event.next_task = next
                        else:
                            poll_stop = rt.poll_time+self.timeout
                            if self.DEBUG:
                                print("Task "+str(rt.task.id)+" will timeout at: "+str(poll_stop))
                            if event.time == -1 or poll_stop < event.time:
                                event.workflow = rt.trace.workflow
                                event.type = 2
                                event.time = poll_stop
                                event.task = rt
                                event.next_task = None
            for trace in self.active_traces:
                poll_stop = trace.poll_time+self.timeout
                if self.DEBUG:
                    print("Trace "+str(trace.id)+" will timeout at: "+str(poll_stop))
                if event.time == -1 or poll_stop < event.time:
                    event.workflow = trace.workflow
                    event.type = 2
                    event.time = poll_stop
                    event.task = trace.initial_task
                    event.next_task = None
            self.forward(event)

            # COLLECT METRICS
            if event.type == 0:
                arrivals[get_service(self.apps[0],event.workflow.initial_task.id)] += 1
            elif event.type == 1:
                if event.next_task is None:
                    times = self.history[event.task.trace.id][event.task.task.id]
                    late_metrics[get_service(self.apps[0],event.task.task.id)] += (times[-1]-times[0])
                    departures[get_service(self.apps[0],event.task.task.id)] += 1
                    if event.task.caller is None:
                        trace_success_metrics[event.workflow.id] += 1
                        total_traces[event.workflow.id] += 1
                else:
                    arrivals[get_service(self.apps[0],event.next_task.id)] += 1
            elif event.type == 2 and event.task.caller is None: # count completed trace only when the initial task times out
                #print("Timeout of trace: "+str(event.task.trace.id))
                trace_fail_metrics[event.workflow.id] += 1
                total_traces[event.workflow.id] += 1
            if self.alloc_method != 0 and it % self.alloc_step_size == 0:
                for i in range(len(self.apps[0].services)):
                    if departures[i] > 0:
                        late_metrics[i] /= departures[i]
                        sum_late_metrics[i] += late_metrics[i]
                        thrput_metrics[i] += arrivals[i] / departures[i]
                        arrival_rate[i] += arrivals[i] / (self.t-prev_step_time)
                        departure_rate[i] += departures[i] / (self.t-prev_step_time)
                    else:
                        late_metrics[i] = 0
                for i in range(len(self.apps[0].workflow)):
                    #print("Successful traces: "+str(trace_success_metrics[i]))
                    #print("Failed traces: "+str(trace_fail_metrics[i]))
                    #print("Total traces: "+str(total_traces[i]))
                    if total_traces[i] > 0:
                        trace_success_metrics[i] = 100*trace_success_metrics[i]/total_traces[i]
                        sum_trace_success[i] += trace_success_metrics[i]
                    else:
                        trace_success_metrics[i] = 0
                prev_step_time = self.t
                if self.alloc_method == 1:
                    res = self.RL_allocation(it,agents,late_metrics,thrput_metrics,trace_success_metrics)
                elif self.alloc_method == 2:
                    res = self.VCG_allocation(arrival_rate,departure_rate)
                for i,s in enumerate(self.apps[0].services):
                    resources[i].append(s.node.cpu)
                if it % self.step_size == 0:
                    for i in range(len(self.apps[0].services)):
                        avg_late_metrics[i].append(sum_late_metrics[i]/(self.step_size/self.alloc_step_size))
                        avg_thrput_metrics[i].append(thrput_metrics[i]/(self.step_size/self.alloc_step_size))
                        avg_arrival_rate[i].append(arrival_rate[i]/(self.step_size/self.alloc_step_size))
                        avg_departure_rate[i].append(departure_rate[i]/(self.step_size/self.alloc_step_size))
                    for i in range(len(self.apps[0].workflows)):
                        avg_trace_success[i].append(sum_trace_success[i]/(self.step_size/self.alloc_step_size))
                    sum_late_metrics = len(self.apps[0].services)*[0]
                    thrput_metrics = len(self.apps[0].services)*[0]
                    arrival_rate = len(self.apps[0].services)*[0]
                    departure_rate = len(self.apps[0].services)*[0]
                    sum_trace_success = len(self.apps[0].workflows)*[0]
                trace_success_metrics = len(self.apps[0].workflows)*[0]
                trace_fail_metrics = len(self.apps[0].workflows)*[0]
                arrivals = len(self.apps[0].services)*[0]
                departures = len(self.apps[0].services)*[0]
                late_metrics = len(self.apps[0].services)*[0]
                total_traces = len(self.apps[0].workflows)*[0]
        """
        for i in range(len(self.apps[0].services)):
            print("SERVICE: "+str(i))
            print("LATENCY")
            plt.plot(range(len(avg_late_metrics[i])), avg_late_metrics[i])
            plt.show()
            print("ARRIVAL RATE")
            plt.plot(range(len(avg_arrival_rate[i])), avg_arrival_rate[i])
            plt.show()
            print("DEPARTURE RATE")
            plt.plot(range(len(avg_departure_rate[i])), avg_departure_rate[i])
            plt.show()
            print("THROUGHPUT")
            plt.plot(range(len(avg_thrput_metrics[i])), avg_thrput_metrics[i])
            plt.show()
            resources_avg = step_average(resources[i],int(self.step_size/self.alloc_step_size))
            print("CPU ALLOCATION")
            plt.plot(range(len(resources_avg)), resources_avg)
            plt.show()
        for i in range(len(self.apps[0].workflows)):
            print("WORKFLOW: "+str(i))
            print("SUCCESS RATE")
            plt.plot(range(len(avg_trace_success[i])), avg_trace_success[i])
            plt.show()
        """

    # RL allocation
    def RL_allocation(self,iter,agents,late_metrics,thrput_metrics,success_rate):
        res = []
        # since we use FCFS policy for obtaining resources, we randomly iterate through the services
        iterator = list(range(len(self.apps[0].services)))
        random.shuffle(iterator)
        for i in iterator:
            s = self.apps[0].services[i]
            latency = sum(late_metrics) / len(late_metrics)
            throughput = sum(thrput_metrics) / len(thrput_metrics)
            success = sum(success_rate) / len(success_rate)
            #agents[i].epsilon = 0.01
            agents[i].epsilon = 1 - iter / self.iterations
            replicas = agents[i].step([min(s.get_queue_size(),QMAX),s.CU.cpu],[latency,throughput,success,s.node.cpu],[LATMIN[i],1.0,100.0,CPUMAX])
            s.CU.node.inc_resources(s.CU,replicas)
            s.set_threads(s.CU.cpu)
            res.append(s.CU.cpu)
        return res

    def VCG_allocation(self,arrival_rate,departure_rate):
        prices = len(self.apps[0].services) * [5]
        quantities = []
        for i in range(len(self.apps[0].services)):
            lamda = arrival_rate[i]
            mu = departure_rate[i]
            prices[i] *= mu
            #print("RATES:")
            #print(lamda)
            #print(mu)
            quantities.append(math.ceil(lamda/mu))
        allocator = VCG_allocator()
        res,payments = allocator.allocate(CPUMAX,quantities,prices)
        #print("ALLOCATION:")
        #print(res)
        for i,s in enumerate(self.apps[0].services):
            s.CU.node.inc_resources(s.CU,res[i])
            s.set_threads(s.CU.cpu)
            res.append(s.CU.cpu)
        return res