import matplotlib

import numpy as np
import matplotlib.pyplot as plt
import ast
from matplotlib.patches import Rectangle

def getMaps(value, policy):
    """
    entries: value dict and policy dict
    return : value_map,Policy_map, the actual Map
    """
    s = np.array(ast.literal_eval(list(value.keys())[0]))
    Value_map = np.empty(s.shape)
    Value_map[:] = np.NaN
    Policy_map = np.chararray(s.shape)
    Policy_map[:] = " "
    Policy_map = Policy_map.astype(str)
    Map  = np.array(ast.literal_eval(list(value.keys())[0]))
    for (k, v) in value.items():
        a,b = np.where(np.array(ast.literal_eval(k))==2)
        Value_map[a,b] = v
        if k in policy:
            Policy_map[a,b] = policy[k]
        A = np.array(ast.literal_eval(k))
        Map = np.where(Map==2, A, Map)
    return Value_map, Policy_map, Map

def drawValuePolicyMap(agent):
    """
    draws policy and value functions of an agent
    """
    actions = {'0' : "S", '1': 'N','2':'W' ,'3': 'E'} # what to write for what action
    Value_map, Policy_map,Map = getMaps(agent.value, agent.policy)
    dim = Map.shape
    fig, ax = plt.subplots()
    fig.set_figheight(dim[0])
    fig.set_figwidth(dim[1])
    im = ax.imshow(Value_map, cmap="RdYlGn")# heat map of value_map


    for i in range(dim[0]):
        for j in range(dim[1]):
            if Map[i,j] == 0:
                if Policy_map[i][j] == '':
                    ax.text(j, i, str("%.2f" % Value_map[i,j]),ha="center", va="center", color="Black")
                else:
                    ax.text(j, i, actions[Policy_map[i][j]],ha="center", va="top", color="Black")
                    ax.text(j, i, str("%.2f" % Value_map[i,j]),ha="center", va="bottom", color="Black")
            elif Map[i][j] == 1:
                ax.add_patch(Rectangle((j-0.5, i-0.5), 1, 1, fill=True, facecolor = "black", edgecolor='black', lw=0))
            elif Map[i][j] == 3:
                text = ax.text(j, i, "Win" ,ha="center", va="center", color="Black")
                ax.add_patch(Rectangle((j-0.5, i-0.5), 1, 1, fill=True, facecolor = "Green", edgecolor='black', lw=0))
            elif Map[i][j] == 4:
                ax.text(j, i, actions[Policy_map[i][j]],ha="center", va="bottom", color="Black")
                text = ax.text(j, i, "Coin" ,ha="center",va="top", color="Black")
                ax.add_patch(Rectangle((j-0.5, i-0.5), 1, 1, fill=True, facecolor = "Yellow", edgecolor='black', lw=0))
            elif Map[i][j] == 5:
                text = ax.text(j, i, "Lose" ,ha="center", va="center", color="Black")
                ax.add_patch(Rectangle((j-0.5, i-0.5), 1, 1, fill=True, facecolor = "red", edgecolor='black', lw=0))
            elif Map[i][j] == 6:
                text = ax.text(j, i, "Rose" ,ha="center", va='bottom', color="Black")
                ax.add_patch(Rectangle((j-0.5, i-0.5), 1, 1, fill=True, facecolor = "pink", edgecolor='black', lw=0))
                ax.text(j, i, str(actions[Policy_map[i][j]]),ha="center", va="top", color="Black")

    return fig
