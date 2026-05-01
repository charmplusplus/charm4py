from os import listdir
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def parsefile(file, data):
    f = open(file, 'r')
    lines = f.readlines()
    length = 0
    time = 0
    for l in lines:
        if "Iterations: " in l:
#             print(l.split())
            length = int(l.split()[-1])
        if "Elapsed Time " in l:
#             print(l.split())
            time = float(l.split()[-2])
            d = {
                'file':file,
                'Iterations':length,
                'Elapsed Time':time
            }
            data = data.append(d,  ignore_index=True)
    return data
  
data = pd.DataFrame(columns =["file", "Iterations", "Elapsed Time"])
data = parsefile("udp-cffi-task-bench.log", data)
data = parsefile("udp-cython-task-bench.log", data)

sns.set_theme(style="whitegrid")
df = data

# Draw a pointplot to show pulse as a function of three categorical factors
g = sns.lineplot(x="Iterations", y="Elapsed Time", hue="file", data=df,ci=99, linewidth=3, markersize=10, alpha  = 0.5)
g.set(ylabel='Elapsed Time')
g.set(xlabel='Iterations')
g.set_xscale('log', base=2)
g.set(title="cffi vs cython with the task-bench benchmark")
plt.subplots_adjust(bottom=0.15)
plt.savefig('cffi_vs_cython_task-bench.png')
print("Created cffi_vs_cython_task-bench.png")
