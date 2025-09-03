import pyspedas
from pytplot import get_data
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, date2num, SecondLocator
import numpy as np
import pandas as pd

# Configurações de tempo e variáveis
start = '2024/03/26 17:11:30'
end = '2024/03/26 17:13:40'
timerange = [start, end]
probes = ['1', '2', '3', '4']
var_fgm = [f'mms{p}_fgm_b_gsm_srvy_l2' for p in probes]
pyspedas.mms.fgm(probe=probes, trange=timerange, varnames=var_fgm, time_clip=True)

# Extrair dados de campo magnético
fgm_data = {}
for p in probes:
    result = get_data(f'mms{p}_fgm_b_gsm_srvy_l2')
    if result:
        timeB, FGM = result
        fgm_data[p] = (pd.to_datetime(timeB, unit='s'), FGM)
        print(f"Dados extraídos para MMS{p}")
    else:
        print(f"Falha ao obter os dados de MMS{p}")

# Configuração do gráfico
fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
colors = ['blue', 'green', 'red', 'darkgoldenrod']
handles, labels = [], []

# Plotar dados de Btotal, Bx, By e Bz
for i, p in enumerate(probes):
    if p in fgm_data:
        time, data = fgm_data[p]
        time_num = date2num(time)
        B_total = np.linalg.norm(data[:, :3], axis=1)
        
        line, = axs[0].plot(time_num, B_total, label=f'MMS{p}', color=colors[i], linewidth=0.75)
        axs[1].plot(time_num, data[:, 0], color=colors[i], linewidth=0.75)
        axs[2].plot(time_num, data[:, 1], color=colors[i], linewidth=0.75)
        axs[3].plot(time_num, data[:, 2], color=colors[i], linewidth=0.75)
        
        handles.append(line)
        labels.append(f'MMS{p}')

# Configurações dos subplots
for ax in axs:
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
axs[0].set_ylabel('Btotal [nT]')
axs[1].set_ylabel('Bx [nT]')
axs[2].set_ylabel('By [nT]')
axs[3].set_ylabel('Bz [nT]')
axs[3].set_xlabel('Tempo [UT]')
axs[3].xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
axs[3].xaxis.set_major_locator(SecondLocator(interval=30))

# Legenda e título
fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.9, 0.5), title='Sondas')
fig.suptitle('Campo Magnético (FGM) - MMS1, MMS2, MMS3, MMS4', x=0.5, y=0.94)
plt.show()
