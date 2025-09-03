import pyspedas
import pandas as pd
from pytplot import tplot, get_data, store_data, options, split_vec
import numpy as np
import matplotlib.pyplot as plt 

start_time = '2024/3/26 17:11:00'
end_time   = '2024/3/26 17:13:45'
trange = [start_time, end_time]
probe = 3
#probe indica o número do satélite MMS

pyspedas.mms.fgm(probe=probe, trange=trange, data_rate='brst', varnames=[f'mms{probe}_fgm_b_gsm_brst_l2'], time_clip=True)
pyspedas.mms.edp(probe=probe, trange=trange, data_rate='brst', varnames=[f'mms{probe}_edp_dce_gse_brst_l2'], time_clip=True)
pyspedas.mms.fpi(probe=probe, trange=trange, data_rate='brst', datatype='dis-moms', varnames=[f'mms{probe}_dis_energyspectr_omni_brst',
                  f'mms{probe}_dis_numberdensity_brst', f'mms{probe}_dis_bulkv_gse_brst'], time_clip=True)
pyspedas.mms.fpi(probe=probe, trange=trange, data_rate='brst', datatype='des-moms', varnames=[f'mms{probe}_des_energyspectr_omni_brst',
                  f'mms{probe}_des_numberdensity_brst', f'mms{probe}_des_bulkv_gse_brst', 
                  f'mms{probe}_des_temppara_brst', f'mms{probe}_des_tempperp_brst'], time_clip=True)
pyspedas.cotrans(name_in=f'mms{probe}_dis_bulkv_gse_brst', name_out=f'mms{probe}_dis_bulkv_gsm_brst', coord_in="gse", coord_out="gsm")
pyspedas.cotrans(name_in=f'mms{probe}_des_bulkv_gse_brst', name_out=f'mms{probe}_des_bulkv_gsm_brst', coord_in="gse", coord_out="gsm")
pyspedas.cotrans(name_in=f'mms{probe}_edp_dce_gse_brst_l2', name_out=f'mms{probe}_edp_dce_gsm_brst_l2', coord_in="gse", coord_out="gsm")
pyspedas.mms.mec(trange=trange, probe=probe, varnames=[f'mms{probe}_mec_r_gsm'], time_clip=True)
pyspedas.mms.state(trange=trange)

transformation_matrix = np.array([[0.02380513, -0.14284547, 0.9894587],
                 [0.06304695, -0.9875545,  -0.14408739],
                 [0.9977266,   0.06581236, -0.01450289]])#(vetores L, M, N como linhas)

timeB, B = get_data(f'mms{probe}_fgm_b_gsm_brst_l2') 
B_gsm = B[:, 0:3]
B_lmn = np.dot(B_gsm, transformation_matrix.T) # Realiza a transformação para LMN
B_total = np.sqrt(np.sum(B_lmn**2, axis=1))  # Calcula o módulo ao longo das linhas
B_combined = np.hstack((B_lmn, B_total[:, None]))  # Adiciona B_total como a 4ª coluna
store_data(f'mms{probe}_fgm_b_gsm_brst_l2_lmn', data={'x': timeB, 'y': B_combined})

timeE, E = get_data(f'mms{probe}_edp_dce_gsm_brst_l2') 
E_lmn = np.dot(E, transformation_matrix.T) 
store_data(f'mms{probe}_edp_dce_gsm_brst_l2_lmn', data={'x': timeE, 'y': E_lmn})

timeVi, Vi = get_data(f'mms{probe}_dis_bulkv_gsm_brst') 
Vi_lmn = np.dot(Vi, transformation_matrix.T) 
store_data(f'mms{probe}_dis_bulkv_gsm_brst_lmn', data={'x': timeVi, 'y': Vi_lmn})

timeVe, Ve = get_data(f'mms{probe}_des_bulkv_gsm_brst') 
Ve_lmn = np.dot(Ve, transformation_matrix.T) 
store_data(f'mms{probe}_des_bulkv_gsm_brst_lmn', data={'x': timeVe, 'y': Ve_lmn})

time_pos, pos = get_data(f'mms{probe}_mec_r_gsm')
pos_re = pos / 6371.2
store_data(f'mms{probe}_mec_r_gsm_re', data={'x': time_pos, 'y': pos_re})
store_data(f'mms{probe}_el_temperatures', data=[f'mms{probe}_des_temppara_brst', f'mms{probe}_des_tempperp_brst'])

q = 1.602e-19  # Constante da carga do elétron (em Coulombs)

pyspedas.tinterpol(names=f'mms{probe}_dis_bulkv_gsm_brst_lmn', interp_to=f'mms{probe}_des_bulkv_gsm_brst_lmn')  # The interpolated data is now stored in 'variable1-itrp'
timeVi, Vi_i = get_data(f'mms{probe}_dis_bulkv_gsm_brst_lmn-itrp')
pyspedas.tinterpol(names=f'mms{probe}_dis_numberdensity_brst', interp_to=f'mms{probe}_des_bulkv_gsm_brst_lmn')
timeNi, N_i = get_data(f'mms{probe}_dis_numberdensity_brst-itrp')

j_lmn = 10e15*(q * N_i[:, None] * (Vi_i - Ve_lmn))  # Calcular densidade de corrente elétrica: J = q * n * (Vi - Ve)
store_data(f'mms{probe}_current_density_lmn', data={'x': timeVe, 'y': j_lmn})

options(f'mms{probe}_fgm_b_gsm_brst_l2_lmn', 'ytitle', 'B')
options(f'mms{probe}_fgm_b_gsm_brst_l2_lmn', 'legend_names', ['BL', 'BM', 'BN', 'Btot'])
options(f'mms{probe}_fgm_b_gsm_brst_l2_lmn', 'legend_size', 8)  # Diminuir o tamanho da legenda

options(f'mms{probe}_edp_dce_gsm_brst_l2_lmn', 'ytitle','E' )
options(f'mms{probe}_edp_dce_gsm_brst_l2_lmn', 'legend_names', ['EL', 'EM', 'EN'])
options(f'mms{probe}_edp_dce_gsm_brst_l2_lmn', 'legend_size', 9) 

options(f'mms{probe}_dis_energyspectr_omni_brst', 'ytitle', 'En i')
options(f'mms{probe}_dis_energyspectr_omni_brst', 'legend_size', 9) 
options(f'mms{probe}_dis_energyspectr_omni_brst', 'axis_font_size', 8) 

options(f'mms{probe}_dis_bulkv_gsm_brst_lmn-itrp', 'ytitle', 'Vi')
options(f'mms{probe}_dis_bulkv_gsm_brst_lmn-itrp', 'legend_names', ['VL', 'VM', 'VN'])
options(f'mms{probe}_dis_bulkv_gsm_brst_lmn-itrp', 'legend_size', 9)  # Diminuir o tamanho da legenda

options(f'mms{probe}_des_bulkv_gsm_brst_lmn', 'ytitle', 'Ve')
options(f'mms{probe}_des_bulkv_gsm_brst_lmn', 'legend_names', ['VL', 'VM', 'VN'])
options(f'mms{probe}_des_bulkv_gsm_brst_lmn', 'legend_size', 9)  # Diminuir o tamanho da legenda

options(f'mms{probe}_el_temperatures', 'ytitle', 'T')
options(f'mms{probe}_el_temperatures', 'Color', ['blue', 'red'])
options(f'mms{probe}_el_temperatures', 'legend_names', ['Tpara', 'Tperp'])
options(f'mms{probe}_el_temperatures', 'legend_size', 9)

options(f'mms{probe}_current_density_lmn', 'ytitle', 'J [µA/m^2]')
options(f'mms{probe}_current_density_lmn', 'ytitle_size', 6 )
options(f'mms{probe}_current_density_lmn', 'legend_names', ['JL', 'JM', 'JN'])
options(f'mms{probe}_current_density_lmn', 'legend_size', 9)

options(f'mms{probe}_dis_numberdensity_brst', 'ytitle', 'Ni')
options(f'mms{probe}_dis_numberdensity_brst', 'ylog', True)

split_vec(f'mms{probe}_mec_r_gsm_re')
options(f'mms{probe}_mec_r_gsm_re_x', 'ytitle', 'X GSM [Re]')
options(f'mms{probe}_mec_r_gsm_re_y', 'ytitle', 'Y GSM [Re]')
options(f'mms{probe}_mec_r_gsm_re_z', 'ytitle', 'Z GSM [Re]')

tplot([ f'mms{probe}_fgm_b_gsm_brst_l2_lmn',f'mms{probe}_edp_dce_gsm_brst_l2_lmn', f'mms{probe}_dis_energyspectr_omni_brst',
        f'mms{probe}_dis_bulkv_gsm_brst_lmn-itrp', f'mms{probe}_des_bulkv_gsm_brst_lmn', f'mms{probe}_el_temperatures', f'mms{probe}_current_density_lmn' ,
        f'mms{probe}_dis_numberdensity_brst'])

options(f'mms{probe}_fgm_b_gsm_brst_l2_lmn', 'legend_size', 12)  # Diminuir o tamanho da legenda
options(f'mms{probe}_edp_dce_gsm_brst_l2_lmn', 'legend_size', 12) 
options(f'mms{probe}_dis_energyspectr_omni_brst', 'legend_size', 12) 
options(f'mms{probe}_dis_bulkv_gsm_brst_lmn-itrp', 'legend_size', 12)  # Diminuir o tamanho da legenda
options(f'mms{probe}_des_bulkv_gsm_brst_lmn', 'legend_size', 12)  # Diminuir o tamanho da legenda
options(f'mms{probe}_el_temperatures', 'legend_size', 12)
options(f'mms{probe}_current_density_lmn', 'legend_size', 12)

tplot(f'mms{probe}_fgm_b_gsm_brst_l2_lmn')
tplot(f'mms{probe}_edp_dce_gsm_brst_l2_lmn')
tplot(f'mms{probe}_dis_energyspectr_omni_brst')
tplot(f'mms{probe}_dis_bulkv_gsm_brst_lmn-itrp')
tplot(f'mms{probe}_des_bulkv_gsm_brst_lmn')
tplot(f'mms{probe}_el_temperatures')
tplot(f'mms{probe}_current_density_lmn')
tplot(f'mms{probe}_dis_numberdensity_brst')

#,var_label=[f'mms{probe}_mec_r_gsm_re_x', f'mms{probe}_mec_r_gsm_re_y', f'mms{probe}_mec_r_gsm_re_z'],xsize=10, ysize=15)

#options(f'mms{probe}_des_energyspectr_omni_fast', 'ytitle', 'En e')
#options(f'mms{probe}_des_energyspectr_omni_fast', 'grid_line_width', 2)  # Aumentar a largura das linhas do grid

#options(f'mms{probe}_des_numberdensity_fast', 'ytitle', 'Ne')
#options(f'mms{probe}_des_numberdensity_fast', 'ylog', True)
#options(f'mms{probe}_fgm_b_gsm_brst_l2_lmn', 'label_size', 7)   # Diminuir o tamanho do texto dos eixos
#options(f'mms{probe}_fgm_b_gsm_brst_l2_lmn', 'grid_line_width', 2)  # Aumentar a largura das linhas do grid
#options(f'mms{probe}_fgm_b_gsm_brst_l2_lmn', 'title_size', 6)  # Diminuir o tamanho do título