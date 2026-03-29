from simulator.utils import *
from simulator.runtime import *
from agents.agents import QLearningAgent


def _sim_qlearning_get_state(self, observations):
    queue_cap = getattr(self, "state_q_cap", 20)
    cpu_cap = getattr(self, "state_cpu_cap", 20)
    queue_size = int(observations.get("queue_size", 0))
    cpu = int(observations.get("cpu", 1))
    return (min(queue_size, queue_cap), min(cpu, cpu_cap))


def _sim_qlearning_reward(self, observations):
    weights = getattr(
        self,
        "reward_weights",
        {"success": 1.0, "latency": 0.01, "queue": 0.05, "throughput": 0.1},
    )
    success = float(observations.get("avg_success_rate", 0.0))
    avg_latency = float(observations.get("avg_latency", 0.0))
    queue_size = float(observations.get("queue_size", 0.0))
    avg_throughput = float(observations.get("avg_throughput", 0.0))
    return (
        weights.get("success", 1.0) * success
        + weights.get("throughput", 0.1) * avg_throughput
        - weights.get("latency", 0.01) * avg_latency
        - weights.get("queue", 0.05) * queue_size
    )


# Bind QLearning defaults in simulator scope so QLearningAgent.step() can run
# without subclassing/overriding abstract methods in agents.py.
QLearningAgent.get_state = _sim_qlearning_get_state
QLearningAgent.reward = _sim_qlearning_reward

class Event():
    def __init__(self,type,time,workflow=None,task=None,next_task=None):
        self.type = type
        self.workflow = workflow
        self.task = task
        self.next_task = next_task
        self.time = time

class Simulation():
    def __init__(self,apps,units,iterations,step_size=None,alloc_step_size=None,timeout=0,alloc_method=0,eval_interval=None,latency_target=None,DEBUG=False):
        self.iterations = iterations
        self.timeout = timeout
        # Legacy allocation parameters are kept for backward compatibility.
        self.step_size = step_size
        self.alloc_step_size = alloc_step_size
        self.alloc_method = alloc_method
        if eval_interval is not None and eval_interval <= 0:
            raise ValueError("eval_interval must be > 0 or None")
        self.eval_interval = eval_interval
        self.latency_target = latency_target
        self.next_eval_time = eval_interval
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
                overhead = getattr(s.node, "thread_cpu_overhead", 0.0)
                effective_threads = total_threads * (1.0 + overhead)
                usage = min(effective_threads, s.node.cpu)
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
        elif event.type == 3:
            if self.DEBUG:
                print("EVALUATION EVENT")
            if self.eval_interval is not None:
                self.next_eval_time += self.eval_interval
        if self.DEBUG:
            print("History:")
            print(self.history)
            for service in self.apps[0].services:
                print("Node "+str(service.node.id)+" active services: "+str([ss.id for ss in service.node.active_services]))
            print("Active traces: "+str([x.id for x in self.active_traces]))

    def _get_agent(self, agents, service_id):
        if agents is None:
            return None
        if isinstance(agents, dict):
            return agents.get(service_id)
        if service_id < len(agents):
            return agents[service_id]
        return None

    def _resolve_node(self, node_ref):
        if node_ref is None:
            return None
        if hasattr(node_ref, "id"):
            return node_ref
        if isinstance(node_ref, int):
            for service in self.apps[0].services:
                if service.node.id == node_ref:
                    return service.node
        return None

    def _move_service(self, service, target_node):
        if target_node is None or service.node == target_node:
            return
        current_node = service.node
        if service in current_node.active_services:
            current_node.active_services.remove(service)
        service.node = target_node
        if len(service.running_tasks) > 0 and service not in target_node.active_services:
            target_node.active_services.append(service)
        if not hasattr(target_node, "mem_metric"):
            target_node.reset()

    def _set_service_cpu(self, service, cpu):
        if cpu is None:
            return
        new_cpu = max(1, int(cpu))
        if hasattr(service, "CU") and hasattr(service.CU, "node"):
            service.CU.node.inc_resources(service.CU, new_cpu)
            service.set_threads(service.CU.cpu)
        else:
            service.set_threads(new_cpu)

    def _apply_agent_action(self, service, action):
        if action is None:
            return
        if isinstance(action, dict):
            node = self._resolve_node(action.get("node"))
            self._move_service(service, node)
            self._set_service_cpu(service, action.get("cpu"))
        elif isinstance(action, (int, float)):
            self._set_service_cpu(service, action)

    def _build_eval_metrics(self, arrivals, departures, late_metrics, trace_success_metrics, trace_fail_metrics, total_traces, elapsed, e2e_latency_sums=None, e2e_counts=None, violation_counts=None):
        services_metrics = []
        for i, service in enumerate(self.apps[0].services):
            dep = departures[i]
            avg_latency = late_metrics[i] / dep if dep > 0 else 0.0
            avg_throughput = arrivals[i] / dep if dep > 0 else 0.0
            arr_rate = arrivals[i] / elapsed if elapsed > 0 else 0.0
            dep_rate = departures[i] / elapsed if elapsed > 0 else 0.0
            services_metrics.append({
                "service_id": service.id,
                "arrivals": arrivals[i],
                "departures": dep,
                "avg_latency": avg_latency,
                "avg_throughput": avg_throughput,
                "arrival_rate": arr_rate,
                "departure_rate": dep_rate,
            })

        workflow_success_rates = []
        for i in range(len(self.apps[0].workflows)):
            success = 100.0 * trace_success_metrics[i] / total_traces[i] if total_traces[i] > 0 else 0.0
            workflow_success_rates.append(success)

        workflow_e2e_latencies = []
        for i in range(len(self.apps[0].workflows)):
            cnt = e2e_counts[i] if e2e_counts is not None else 0
            avg_e2e = (e2e_latency_sums[i] / cnt) if (e2e_latency_sums is not None and cnt > 0) else 0.0
            workflow_e2e_latencies.append(avg_e2e)

        workflow_violation_rates = []
        for i in range(len(self.apps[0].workflows)):
            cnt = e2e_counts[i] if e2e_counts is not None else 0
            vio = violation_counts[i] if violation_counts is not None else 0
            rate = (vio / cnt) if cnt > 0 else 0.0
            workflow_violation_rates.append(rate)

        accumulated = {
            "services": services_metrics,
            "workflows": {
                "success": trace_success_metrics.copy(),
                "failed": trace_fail_metrics.copy(),
                "total": total_traces.copy(),
                "success_rate": workflow_success_rates,
                "avg_success_rate": sum(workflow_success_rates) / len(workflow_success_rates) if len(workflow_success_rates) > 0 else 0.0,
                "e2e_latencies": workflow_e2e_latencies,
                "violation_rates": workflow_violation_rates,
                "avg_violation_rate": sum(workflow_violation_rates) / len(workflow_violation_rates) if len(workflow_violation_rates) > 0 else 0.0,
            },
            "elapsed": elapsed,
        }
        instant = {
            "services": [
                {
                    "service_id": s.id,
                    "queue_size": s.get_queue_size(),
                    "running_tasks": len(s.running_tasks),
                    "cpu": getattr(s, "CU", s).cpu if hasattr(getattr(s, "CU", s), "cpu") else s.threads,
                    "threads": s.threads,
                    "node_id": s.node.id,
                }
                for s in self.apps[0].services
            ],
        }
        return accumulated, instant

    def run(self, agents=None):
        self.t = 0
        self.next_eval_time = self.eval_interval
        prev_eval_time = 0

        arrivals = len(self.apps[0].services) * [0]
        departures = len(self.apps[0].services) * [0]
        late_metrics = len(self.apps[0].services) * [0]
        total_traces = len(self.apps[0].workflows) * [0]
        trace_success_metrics = len(self.apps[0].workflows) * [0]
        trace_fail_metrics = len(self.apps[0].workflows) * [0]
        e2e_latency_sums = len(self.apps[0].workflows) * [0.0]
        e2e_counts = len(self.apps[0].workflows) * [0]
        violation_counts = len(self.apps[0].workflows) * [0]

        for _ in range(1, self.iterations + 1):
            event = Event(0, -1)

            if self.DEBUG:
                print("\n\n######################################################################\n")
                print("PREVIOUS TIME: " + str(self.t))

            for app in self.apps:
                for tt in app.workflows:
                    if self.DEBUG:
                        print("New trace from workflow " + str(tt.nodes) + " at time: " + str(tt.next_arrival))
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
                            estim, next_task = rt.estimation(speed, self.t)
                            if self.DEBUG:
                                print("Task " + str(rt.task.id) + " will end or call another task in trace: " + str(rt.trace.id) + " --> " + str(estim))
                            if event.time == -1 or estim < event.time:
                                event.workflow = rt.trace.workflow
                                event.type = 1
                                event.time = estim
                                event.task = rt
                                event.next_task = next_task
                        else:
                            poll_stop = rt.poll_time + self.timeout
                            if self.DEBUG:
                                print("Task " + str(rt.task.id) + " will timeout at: " + str(poll_stop))
                            if event.time == -1 or poll_stop < event.time:
                                event.workflow = rt.trace.workflow
                                event.type = 2
                                event.time = poll_stop
                                event.task = rt
                                event.next_task = None

            for trace in self.active_traces:
                poll_stop = trace.poll_time + self.timeout
                if self.DEBUG:
                    print("Trace " + str(trace.id) + " will timeout at: " + str(poll_stop))
                if event.time == -1 or poll_stop < event.time:
                    event.workflow = trace.workflow
                    event.type = 2
                    event.time = poll_stop
                    event.task = trace.initial_task
                    event.next_task = None

            if self.next_eval_time is not None and (event.time == -1 or self.next_eval_time < event.time):
                event.type = 3
                event.time = self.next_eval_time
                event.workflow = None
                event.task = None
                event.next_task = None

            self.forward(event)

            if event.type == 0:
                arrivals[get_service(self.apps[0], event.workflow.initial_task.id)] += 1
            elif event.type == 1:
                if event.next_task is None:
                    times = self.history[event.task.trace.id][event.task.task.id]
                    late_metrics[get_service(self.apps[0], event.task.task.id)] += (times[-1] - times[0])
                    departures[get_service(self.apps[0], event.task.task.id)] += 1
                    if event.task.caller is None:
                        e2e_latency = times[-1] - times[0]
                        trace_success_metrics[event.workflow.id] += 1
                        total_traces[event.workflow.id] += 1
                        e2e_latency_sums[event.workflow.id] += e2e_latency
                        e2e_counts[event.workflow.id] += 1
                        if self.latency_target is not None and e2e_latency > self.latency_target:
                            violation_counts[event.workflow.id] += 1
                else:
                    arrivals[get_service(self.apps[0], event.next_task.id)] += 1
            elif event.type == 2 and event.task.caller is None:
                trace_fail_metrics[event.workflow.id] += 1
                total_traces[event.workflow.id] += 1
            elif event.type == 3:
                elapsed = self.t - prev_eval_time
                accumulated_metrics, instant_metrics = self._build_eval_metrics(
                    arrivals,
                    departures,
                    late_metrics,
                    trace_success_metrics,
                    trace_fail_metrics,
                    total_traces,
                    elapsed,
                    e2e_latency_sums,
                    e2e_counts,
                    violation_counts,
                )

                if agents is not None:
                    for i, service in enumerate(self.apps[0].services):
                        agent = self._get_agent(agents, service.id)
                        if agent is None:
                            continue
                        if hasattr(agent, "on_eval"):
                            action = agent.on_eval(i, service, accumulated_metrics, instant_metrics)
                        elif callable(agent):
                            action = agent(i, service, accumulated_metrics, instant_metrics)
                        else:
                            continue
                        self._apply_agent_action(service, action)

                prev_eval_time = self.t
                trace_success_metrics = len(self.apps[0].workflows) * [0]
                trace_fail_metrics = len(self.apps[0].workflows) * [0]
                arrivals = len(self.apps[0].services) * [0]
                departures = len(self.apps[0].services) * [0]
                late_metrics = len(self.apps[0].services) * [0]
                total_traces = len(self.apps[0].workflows) * [0]
                e2e_latency_sums = len(self.apps[0].workflows) * [0.0]
                e2e_counts = len(self.apps[0].workflows) * [0]
                violation_counts = len(self.apps[0].workflows) * [0]