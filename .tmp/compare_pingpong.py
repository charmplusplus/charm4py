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
        if "RES:: [plain] " in l:
            length = int(l.split()[2])
            time = float(l.split()[-1])
            d = {
                'file':file,
                'msg length':length,
                'Elapsed Time':time
            }
            data = data.append(d,  ignore_index=True)
    return data
  
data = pd.DataFrame(columns =["file", "msg length", "Elapsed Time"])
data = parsefile("udp-cffi-pingpong.log", data)
data = parsefile("udp-cython-pingpong.log", data)

sns.set_theme(style="whitegrid")
df = data

# Draw a pointplot to show pulse as a function of three categorical factors
g = sns.lineplot(x="msg length", y="Elapsed Time", hue="file", data=df,ci=99, linewidth=3, markersize=10, alpha  = 0.5)
g.set(ylabel='Elapsed Time')
g.set(xlabel='msg length (ms)')
g.set_xscale('log', base=2)
g.set(title="cffi vs cython with the pingpong benchmark")
plt.subplots_adjust(bottom=0.15)
plt.savefig('cffi_vs_cython_pingpong.png')
print("Created cffi_vs_cython_pingpong.png")
