# Módulos
import pyspedas
import pandas as pd
from pytplot import tplot, get_data, store_data, options, split_vec
from pytplot.tplot_math import time_clip as tclip
import numpy as np
import matplotlib.pyplot as plt

#----------------------------------------------------------------------------------------------------
# Data
probe = 1
Start_mean = '2024/3/26 17:08:00'
End_mean = '2024/3/26 17:16:00'
trangetot= [Start_mean, End_mean]
pyspedas.mms.fpi(probe=probe, trange=trangetot, data_rate='brst', datatype='dis-moms', 
                 varnames=[ f'mms{probe}_dis_bulkv_gse_brst'], time_clip=True)
pyspedas.cotrans(name_in=f'mms{probe}_dis_bulkv_gse_brst', name_out=f'mms{probe}_dis_bulkv_gsm_brst', coord_in="gse", coord_out="gsm")

transformation_matrix = np.array([[0.03086242, -0.16024368, 0.9865949],
                                [0.06529449, -0.9846335, -0.16196764],
                                [0.99738866,  0.06941792, -0.01992514]])#(vetores L, M, N como linhas)

timeVi, Vi = get_data(f'mms{probe}_dis_bulkv_gsm_brst') 
Vi_lmn = np.dot(Vi, transformation_matrix.T) 
store_data(f'mms{probe}_dis_bulkv_lmn_brst', data={'x': timeVi, 'y': Vi_lmn})

options(f'mms{probe}_dis_bulkv_lmn_brst', 'ytitle', 'V ion')
options(f'mms{probe}_dis_bulkv_lmn_brst', 'ysubtitle', '[km/s]')
options(f'mms{probe}_dis_bulkv_lmn_brst','legend_names', ['VL','VM','VN'])
options(f'mms{probe}_dis_bulkv_lmn_brst', 'legend_size', 12) 
options(f'mms{probe}_dis_bulkv_lmn_brst', 'title_size', 20)   
tplot(f'mms{probe}_dis_bulkv_lmn_brst')


#----------------------------------------------------------------------------------------------------
# Time
start_timeSH = '2024/3/26 17:13:15'
end_timeSH  = '2024/3/26 17:13:30'
trangeSH = [start_timeSH, end_timeSH]

start_timeVmax = '2024/3/26 17:11:50'
end_timeVmax = '2024/3/26 17:13:00'
trangeV = [start_timeVmax, end_timeVmax]

#----------------------------------------------------------------------------------------------------

# 1- V outflow
Vsplit = f'mms{probe}_dis_bulkv_lmn_brst'

# Achar o instante Vz máximo
tclip(Vsplit, time_start=start_timeVmax, time_end=end_timeVmax, suffix='_outflow')
result = get_data(Vsplit + '_outflow')

if result is not None:
    t_outflow, VarV_outflow = result

    # Separar componentes
    VN_outflow = VarV_outflow[:, 0]
    VM_outflow = VarV_outflow[:, 1]
    VL_outflow = VarV_outflow[:, 2]

    # Encontrar índice do valor máximo de Vz
    index_max_z = np.argmax(np.abs(VL_outflow))

    # Obter valores correspondentes de Vx e Vy
    vN_max = VN_outflow[index_max_z]
    vM_max = VM_outflow[index_max_z]
    vL_max = VL_outflow[index_max_z]

    Vmaxdata = np.array([[vN_max, vM_max, vL_max]])

    # V1 = Vsplit
    tclip(Vsplit, time_start=start_timeSH, time_end=end_timeSH, suffix='sHeath')
    tsHeath, V1sHeath = get_data(Vsplit + 'sHeath')
    av_V1 = np.nanmean(V1sHeath, axis=0)

    Vsh = av_V1

    Voutflow = Vmaxdata - Vsh

    print('Vmax:', Vmaxdata)
    print('Vsh:', Vsh)
    print('Vout:', Voutflow)
else:
    print("Não foi possível obter os dados de Vmax_Z")
