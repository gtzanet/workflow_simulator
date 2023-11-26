from matplotlib import pyplot as plt
import networkx as nx

colors = ["blue","red","green","orange","pink","purple","brown","yellow","black"]

def draw_graph(G,color_features,name):
    fig, ax = plt.subplots(figsize=(5, 5))
    color_map = []
    for n in color_features:
        color_map.append(colors[n])
    #plt.figure(name)
    nx.draw(G, node_color=color_map, with_labels = True,node_size=500,font_size=15)
    plt.show()

def moving_average(l,window_size):
    if not l:
        return []
    res = []
    for i in range(len(l) - window_size + 1):
        window = l[i : i + window_size]
        res.append(sum(window) / window_size)
    return res

def step_average(l,window_size):
    res = []
    for i in range((len(l)-1) // window_size + 1):
        window = l[i*window_size : (i+1)*window_size]
        res.append(sum(window) / len(window))
    return res

def get_service(app,task_id):
    return app.task_graph.nodes[task_id]["subset"]