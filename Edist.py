import pyspedas
from pytplot import tplot, get_data, store_data, options, split_vec
import numpy as np
import matplotlib.pyplot as plt 
#-----------------------TIME-------------------------------------
start_time = '2024/3/26 17:10:30'
end_time   = '2024/3/26 17:13:45'
trange = [start_time, end_time]
probe = 1
#-----------------------LOAD VAR-------------------------------------
datarate_e = 'fast'
coord='gsm'
pyspedas.mms.fgm(probe=probe, trange=trange, data_rate='brst', varnames=[f'mms{probe}_fgm_b_'+coord+'_brst_l2'], time_clip=True)
pyspedas.mms.edp(probe=probe, trange=trange, data_rate=datarate_e, varnames=[f'mms{probe}_edp_dce_gse_'+datarate_e+'_l2'], time_clip=True)
pyspedas.mms.fpi(probe=probe, trange=trange, data_rate='brst', datatype='dis-moms', varnames=[f'mms{probe}_dis_energyspectr_omni_brst',
                  f'mms{probe}_dis_numberdensity_brst', f'mms{probe}_dis_bulkv_gse_brst'], time_clip=True)
pyspedas.mms.fpi(probe=probe, trange=trange, data_rate='brst', datatype='des-moms', varnames=[f'mms{probe}_des_energyspectr_omni_brst',
                  f'mms{probe}_des_numberdensity_brst', f'mms{probe}_des_bulkv_gse_brst', 
                  f'mms{probe}_des_temppara_brst', f'mms{probe}_des_tempperp_brst'], time_clip=True)
pyspedas.cotrans(name_in=f'mms{probe}_dis_bulkv_gse_brst', name_out=f'mms{probe}_dis_bulkv_gsm_brst', coord_in="gse", coord_out="gsm")
pyspedas.cotrans(name_in=f'mms{probe}_des_bulkv_gse_brst', name_out=f'mms{probe}_des_bulkv_gsm_brst', coord_in="gse", coord_out="gsm")
pyspedas.cotrans(name_in=f'mms{probe}_edp_dce_gse_'+datarate_e+'_l2', name_out=f'mms{probe}_edp_dce_gsm_'+datarate_e+'_l2', coord_in="gse", coord_out="gsm")
pyspedas.mms.mec(trange=trange, probe=probe, varnames=[f'mms{probe}_mec_r_gsm'], time_clip=True)
pyspedas.mms.state(trange=trange)

#-----------------------LMN ROTATION----------------------------------------
transformation_matrix = np.array([[0.03086242, -0.16024368, 0.9865949],
                 [0.06529449, -0.9846335, -0.16196764],
                 [0.99738866,  0.06941792, -0.01992514]])
#-----------------------CALC----------------------------------------
#B
timeB, B = get_data(f'mms{probe}_fgm_b_'+coord+'_brst_l2') 
B_gsm = B[:, 0:3]
B_lmn = np.dot(B_gsm, transformation_matrix.T) # Realiza a transformação para LMN
store_data(f'mms{probe}_fgm_b_'+coord+'_brst_l2_lmn_vec', data={'x': timeB, 'y': B_lmn})
B_lmn_vec = get_data(f'mms{probe}_fgm_b_'+coord+'_brst_l2_lmn_vec')[1]  # Aqui, pegamos os dados reais e não o nome da variável
B_total = np.sqrt(np.sum(B_lmn**2, axis=1))  # Calcula o módulo ao longo das linhas
B_combined = np.hstack((B_lmn, B_total[:, None]))  # Adiciona B_total como a 4ª coluna
store_data(f'mms{probe}_fgm_b_'+coord+'_brst_l2_lmn', data={'x': timeB, 'y': B_combined})

#E
timeE, E = get_data(f'mms{probe}_edp_dce_'+coord+'_'+datarate_e+'_l2') 
E_lmn = np.dot(E, transformation_matrix.T) 
store_data(f'mms{probe}_edp_dce_'+coord+'_'+datarate_e+'_l2_lmn', data={'x': timeE, 'y': E_lmn})

#Vi e Ve
timeVi, Vi = get_data(f'mms{probe}_dis_bulkv_'+coord+'_brst') 
Vi_lmn = np.dot(Vi, transformation_matrix.T) 
store_data(f'mms{probe}_dis_bulkv_'+coord+'_brst_lmn', data={'x': timeVi, 'y': Vi_lmn})
timeVe, Ve = get_data(f'mms{probe}_des_bulkv_'+coord+'_brst') 
Ve_lmn = np.dot(Ve, transformation_matrix.T) 
store_data(f'mms{probe}_des_bulkv_'+coord+'_brst_lmn', data={'x': timeVe, 'y': Ve_lmn})

#R gse
time_pos, pos = get_data(f'mms{probe}_mec_r_'+coord)
pos_re = pos / 6371.2
store_data(f'mms{probe}_mec_r_'+coord+'_re', data={'x': time_pos, 'y': pos_re})
store_data(f'mms{probe}_el_temperatures', data=[f'mms{probe}_des_temppara_brst', f'mms{probe}_des_tempperp_brst'])

ve_norma = np.sqrt(Ve[:,0]**2+Ve[:,1]**2+Ve[:,2]**2)
store_data(f've_normalizada', data={'x':timeVe,'y':ve_norma})
vi_norma = np.sqrt(Vi[:,0]**2+Vi[:,1]**2+Vi[:,2]**2)
store_data(f'vi_normalizada', data={'x':timeVi,'y':vi_norma})
b_versor = B_lmn / B_total[:, np.newaxis]

q = 1.602*(10**(-19))  # Constante da carga do elétron (em Coulombs)

#-----------------------INTERP----------------------------------------
pyspedas.tinterpol(names=f'mms{probe}_edp_dce_'+coord+'_'+datarate_e+'_l2_lmn', interp_to=f'mms{probe}_fgm_b_'+coord+'_brst_l2_lmn_vec') 
timeE, E_i = get_data(f'mms{probe}_edp_dce_'+coord+'_'+datarate_e+'_l2_lmn-itrp')
pyspedas.tinterpol(names=f'mms{probe}_des_bulkv_'+coord+'_brst_lmn', interp_to=f'mms{probe}_fgm_b_'+coord+'_brst_l2_lmn_vec') 
timeVe, Ve_i = get_data(f'mms{probe}_des_bulkv_'+coord+'_brst_lmn-itrp') # The interpolated data is now stored in 'variable1-itrp'
pyspedas.tinterpol(names=f'mms{probe}_dis_bulkv_'+coord+'_brst_lmn', interp_to=f'mms{probe}_fgm_b_'+coord+'_brst_l2_lmn_vec')  
timeVi, Vi_i = get_data(f'mms{probe}_dis_bulkv_'+coord+'_brst_lmn-itrp')
pyspedas.tinterpol(names=f'mms{probe}_dis_numberdensity_brst', interp_to=f'mms{probe}_fgm_b_'+coord+'_brst_l2_lmn_vec')
timeNi, N_i = get_data(f'mms{probe}_dis_numberdensity_brst-itrp')
pyspedas.tinterpol(names=f'mms{probe}_des_numberdensity_brst', interp_to=f'mms{probe}_fgm_b_'+coord+'_brst_l2_lmn_vec')
timeNe, N_e = get_data(f'mms{probe}_des_numberdensity_brst-itrp')
pyspedas.tinterpol(names=f've_normalizada', interp_to=f'mms{probe}_fgm_b_'+coord+'_brst_l2_lmn_vec')
timeVe, Ve_norm = get_data(f've_normalizada-itrp')
pyspedas.tinterpol(names=f'vi_normalizada', interp_to=f'mms{probe}_fgm_b_'+coord+'_brst_l2_lmn_vec')
timeVi, Vi_norm = get_data(f'vi_normalizada-itrp')
#--------------------------------------------------------------------------------------------

#Veper e Viper
escalarVe = Ve_i[:,0]*(B_lmn[:,0]/B_total) + Ve_i[:,1]*(B_lmn[:,1]/B_total)  + Ve_i[:,2]*(B_lmn[:,2]/B_total) 
escalarVi = Vi_i[:,0]*(B_lmn[:,0]/B_total)  + Vi_i[:,1]*(B_lmn[:,1]/B_total)  + Vi_i[:,2]*(B_lmn[:,2]/B_total) 
ve_per = np.sqrt((Ve_norm)**2 - (escalarVe)**2)#ve_perp = sqrt(ve^2 - (ve*b)^2)
store_data(f've_per_plot', data={'x':timeVe,'y':ve_per})
vi_per = np.sqrt((Vi_norm)**2 - (escalarVi)**2)#vi_perp = sqrt(vi^2 - (vi*b)^2)
store_data(f'vi_per_plot', data={'x':timeVi,'y':vi_per})
store_data(f'mms{probe}v_perp', data=[f'vi_per_plot', f've_per_plot'])

#J, E' e J.E'
j_lmn = (1e15)*(q *N_e[:,None]*(Vi_i - Ve_i))   # Calcular densidade de corrente elétrica: J = q * n * (Vi - Ve)
store_data(f'mms{probe}_current_density_lmn', data={'x': timeVe, 'y': j_lmn})
E1 = E_i + np.cross(Ve_i, B_lmn_vec)/1000 #E1 corresponde a E' 
store_data(f'mms{probe}_calculated_Eline', data={'x': timeVe, 'y': E1})
JE = np.sum(E1*j_lmn, axis=1)  
store_data(f'mms{probe}_JE', data={'x': timeVe, 'y': JE})

#Epara
E_para = np.sum(E_i* b_versor, axis=1)  #E_i é o valor de E GSE interpolado 
E_para1 = store_data(f'mms{probe}_E_para', data={'x': timeVe, 'y': E_para})


#-(VexB) e Eperp M
#E_total = np.sqrt(E_i[:,0]**2 +E_i[:,1]**2 +E_i[:,2]**2)
E_perp_L = E_i[:,0] - E_para[:]*b_versor[:,0]
E_perp_M = E_i[:,1] - E_para[:]*b_versor[:,1]
E_perp_N = E_i[:,2] - E_para[:]*b_versor[:,2]
E_perp = np.sqrt(E_perp_L**2 + E_perp_M**2+ E_perp_N**2)        
store_data(f'mms{probe}_E_perp', data={'x': timeVe, 'y': E_perp_M})
VexB= -1*np.cross(Ve_i, B_lmn)/1000
VexB_M = VexB[:, 1]
store_data(f'-(VexB)m', data={'x': timeVe, 'y': VexB_M})
store_data(f'mms{probe}_Eperp_m', data=[f'-(VexB)m',f'mms{probe}_E_perp'])

#E'para = E'xBversor
#E'perp = sqrt(E'^2-E'par^2)

#Jpara*E'para
E1para = np.sum(E1*b_versor, axis=1)
Jpara = np.sum(j_lmn*b_versor, axis=1)
J_para1 = store_data(f'mms{probe}_J_para1', data={'x': timeVe, 'y': Jpara})
JE1para = E_para*Jpara 
store_data(f'mms{probe}_JE1_para',data={'x':timeVe,'y':JE1para})

#Jperp*E'perp
E1perp_L = E1[:,0] - E1para[:]*b_versor[:,0]
E1perp_M = E1[:,1] - E1para[:]*b_versor[:,1]
E1perp_N = E1[:,2] - E1para[:]*b_versor[:,2]
E1perp = np.sqrt(E1perp_L**2 + E1perp_M**2 + E1perp_N**2)
Jperp_L = j_lmn[:,0] - Jpara[:]*b_versor[:,0]
Jperp_M = j_lmn[:,1] - Jpara[:]*b_versor[:,1]
Jperp_N = j_lmn[:,2] - Jpara[:]*b_versor[:,2]
JpeT= np.sqrt(Jperp_L**2+Jperp_M**2+Jperp_N**2)
J_perp1 = store_data(f'mms{probe}_J_perp1', data={'x': timeVe, 'y': JpeT})
JE1perp = (E1perp_L*Jperp_L + E1perp_N*Jperp_N + E1perp_M*Jperp_M)
store_data(f'mms{probe}_JE1_perp',data={'x':timeVe,'y':JE1perp})
store_data(f"mms{probe}_J'E_3", data=[f'mms{probe}_JE',f'mms{probe}_JE1_perp',f'mms{probe}_JE1_para'])

#q = 1.602*(10**(-19))  Constante da carga do elétron (em Coulombs)
C = 299792458 #m/s
meletron = 9.109*(10**(-31)) #kg
E0 = 8.85*(10**(-12)) #Farads/metro(F/m)
Ne_si = N_e*(10**(-6)) #partículas/cm^-3 para m^-3
DE = (C/(np.sqrt(Ne_si*(q**2)/(meletron*E0))))/1000 #divide por 1000 para sair em km
store_data(f'mms{probe}_DE',data={'x':timeVe,'y':DE})

mion = 1.673*10**(-27)#kg - massa do proton


#-----------------------CONFIG PLOT----------------------------------------
legend_size = 10

options(f'mms{probe}_fgm_b_'+coord+'_brst_l2_lmn', 'ytitle', 'B')
options(f'mms{probe}_fgm_b_'+coord+'_brst_l2_lmn', 'ysubtitle',  '[nT]')
options(f'mms{probe}_fgm_b_'+coord+'_brst_l2_lmn', 'legend_names', ['BL', 'BM', 'BN', 'Btot'])
options(f'mms{probe}_fgm_b_'+coord+'_brst_l2_lmn', 'legend_size', legend_size)
options(f'mms{probe}_fgm_b_'+coord+'_brst_l2_lmn', 'legend_location','upper left')
#options(f'mms{probe}_fgm_b_'+coord+'_brst_l2_lmn', 'yrange',[-25,25]) 

options(f'mms{probe}_dis_energyspectr_omni_brst', 'ytitle', 'Ions')
options(f'mms{probe}_des_energyspectr_omni_brst', 'ytitle', 'Electrons')

options(f'mms{probe}_edp_dce_'+coord+'_'+datarate_e+'_l2_lmn-itrp', 'ytitle','E' )
options(f'mms{probe}_edp_dce_'+coord+'_'+datarate_e+'_l2_lmn-itrp', 'ysubtitle',  '[mV/m]' )
options(f'mms{probe}_edp_dce_'+coord+'_'+datarate_e+'_l2_lmn-itrp', 'legend_names', ['EL', 'EM', 'EN'])
options(f'mms{probe}_edp_dce_'+coord+'_'+datarate_e+'_l2_lmn-itrp', 'legend_size', legend_size) 
#options(f'mms{probe}_edp_dce_'+coord+'_'+datarate_e+'_l2_lmn-itrp', 'yrange',[-20,20]) 

options(f'mms{probe}_E_para', 'ytitle','E para' )
options(f'mms{probe}_E_para', 'ysubtitle',  '[mV/m]' )
options(f'mms{probe}_E_para', 'legend_names', ['E para'])
options(f'mms{probe}_E_para', 'legend_size', legend_size) 
#options(f'mms{probe}_E_para', 'yrange',[-20,20])

options(f'mms{probe}_dis_bulkv_'+coord+'_brst_lmn-itrp', 'ytitle', 'Vi')
options(f'mms{probe}_dis_bulkv_'+coord+'_brst_lmn-itrp','ysubtitle',  '[km/s]')
options(f'mms{probe}_dis_bulkv_'+coord+'_brst_lmn-itrp', 'legend_names', ['ViL', 'ViM', 'ViN'])
options(f'mms{probe}_dis_bulkv_'+coord+'_brst_lmn-itrp', 'legend_size', legend_size)
options(f'mms{probe}_dis_bulkv_'+coord+'_brst_lmn-itrp' ,'legend_location','upper left')  # Diminuir o tamanho da legenda
#options(f'mms{probe}_dis_bulkv_'+coord+'_brst_lmn-itrp', 'yrange', [-500,200]) 

options(f'mms{probe}_des_bulkv_'+coord+'_brst_lmn-itrp', 'ytitle', 'Ve')
options(f'mms{probe}_des_bulkv_'+coord+'_brst_lmn-itrp','ysubtitle',  '[km/s]')
options(f'mms{probe}_des_bulkv_'+coord+'_brst_lmn-itrp', 'legend_names', ['VL', 'VM', 'VN'])
options(f'mms{probe}_des_bulkv_'+coord+'_brst_lmn-itrp', 'legend_size', legend_size)  # Diminuir o tamanho da legenda

options(f'mms{probe}v_perp', 'ytitle', 'V perp ')
options(f'mms{probe}v_perp','ysubtitle',  '[km/s]')
options(f'mms{probe}v_perp', 'Color', ['blue', 'red'])
options(f'mms{probe}v_perp', 'legend_names', ['Viperp', 'Veperp'])
options(f'mms{probe}v_perp', 'legend_size', legend_size)
#options(f'mms{probe}v_perp', 'yrange', [0,1500])

options(f'mms{probe}_Eperp_m', 'ytitle', 'E perp ')
options(f'mms{probe}_Eperp_m','ysubtitle',  '[mV/m]')
options(f'mms{probe}_Eperp_m', 'Color', ['red', 'black'])
options(f'mms{probe}_Eperp_m', 'legend_names', ['-(VexB)', 'Eperp M'])
options(f'mms{probe}_Eperp_m', 'legend_size', legend_size)
#options(f'mms{probe}_Eperp_m', 'yrange', [-20,20])

options(f'mms{probe}_el_temperatures', 'ytitle', 'T')
options(f'mms{probe}_el_temperatures','ysubtitle',  '[eV]')
options(f'mms{probe}_el_temperatures', 'Color', ['blue', 'red'])
options(f'mms{probe}_el_temperatures', 'legend_names', ['Tpara', 'Tperp'])
options(f'mms{probe}_el_temperatures', 'legend_size', legend_size)

options(f'mms{probe}_current_density_lmn', 'ytitle', 'J')
options(f'mms{probe}_current_density_lmn','ysubtitle',  '[µA/m^2]')
options(f'mms{probe}_current_density_lmn', 'ytitle_size', 6 )
options(f'mms{probe}_current_density_lmn', 'legend_names', ['JL', 'JM', 'JN'])
options(f'mms{probe}_current_density_lmn', 'legend_size', legend_size)
options(f'mms{probe}_current_density_lmn', 'legend_location','upper left') 


options(f'mms{probe}_JE', 'ytitle', "J.E'")
options(f'mms{probe}_JE', 'ysubtitle', '[nW/m^3]')
options(f'mms{probe}_JE', 'ytitle_size', 6 )
options(f'mms{probe}_JE', 'legend_names', ["J.E'"])
options(f'mms{probe}_JE', 'legend_size', legend_size)

options(f"mms{probe}_J'E_3", 'ytitle', "J.E'")
options(f"mms{probe}_J'E_3", 'ysubtitle', '[nW/m^3]')
options(f"mms{probe}_J'E_3", 'ytitle_size', 6 )
options(f"mms{probe}_J'E_3", 'Color', ['black', 'red', 'blue'])
options(f"mms{probe}_J'E_3", 'legend_names', ["J.E'","Jpe.E'pe","Jpa.E'pa"])
options(f"mms{probe}_J'E_3", 'legend_size', legend_size)

options(f'mms{probe}_calculated_Eline', 'ytitle', 'E*')
options(f'mms{probe}_calculated_Eline', 'ysubtitle', '[mV/m]')
options(f'mms{probe}_calculated_Eline', 'ytitle_size', 6 )
options(f'mms{probe}_calculated_Eline', 'legend_names', ['E*L', 'E*M', 'E*N'])
options(f'mms{probe}_calculated_Eline', 'legend_size', legend_size)

options(f'mms{probe}_des_numberdensity_brst-itrp', 'ytitle', 'Ne')
options(f'mms{probe}_des_numberdensity_brst-itrp', 'ysubtitle', '[cm^-3]')
options(f'mms{probe}_des_numberdensity_brst-itrp', 'ylog', False)

options(f'mms{probe}_DE', 'ytitle', 'DE')
options(f'mms{probe}_DE', 'ysubtitle',  '[km]')
#options(f'mms{probe}_DE', 'yrange', [-1000000,0])

split_vec(f'mms{probe}_mec_r_'+coord+'_re')
options(f'mms{probe}_mec_r_'+coord+'_re_x', 'ytitle', 'X GSM [Re]')
options(f'mms{probe}_mec_r_'+coord+'_re_y', 'ytitle', 'Y GSM [Re]')
options(f'mms{probe}_mec_r_'+coord+'_re_z', 'ytitle', 'Z GSM [Re]')

#------------------------PLOT-------------------------------
#tplot([f'mms{probe}_fgm_b_'+coord+'_brst_l2_lmn'])
#plot([f'mms{probe}_dis_bulkv_'+coord+'_brst_lmn-itrp',f'mms{probe}v_perp'])
#tplot([f'mms{probe}_el_temperatures'])
#tplot([f'mms{probe}_current_density_lmn'])
#tplot([f'mms{probe}_edp_dce_'+coord+'_'+datarate_e+'_l2_lmn-itrp'])
#tplot([f"mms{probe}_J'E_3"])

tplot([f'mms{probe}_fgm_b_'+coord+'_brst_l2_lmn',
       f'mms{probe}_dis_energyspectr_omni_brst',
       f'mms{probe}_dis_bulkv_'+coord+'_brst_lmn-itrp',
       f'mms{probe}_dis_energyspectr_omni_brst',
       f'mms{probe}_current_density_lmn'
       ])

tplot([f'mms{probe}_fgm_b_'+coord+'_brst_l2_lmn',
       f'mms{probe}_dis_energyspectr_omni_brst',
       #f'mms{probe}_des_energyspectr_omni_brst',
       f'mms{probe}_des_numberdensity_brst-itrp',
       f'mms{probe}_dis_bulkv_'+coord+'_brst_lmn-itrp',
       f'mms{probe}v_perp',
       f'mms{probe}_current_density_lmn',
       f'mms{probe}_el_temperatures',
       f'mms{probe}_edp_dce_'+coord+'_'+datarate_e+'_l2_lmn-itrp'
])

tplot([f'mms{probe}_fgm_b_'+coord+'_brst_l2_lmn',
       f'mms{probe}_current_density_lmn',
       f'mms{probe}_edp_dce_'+coord+'_'+datarate_e+'_l2_lmn-itrp',
       f'mms{probe}_Eperp_m',
       f'mms{probe}_E_para',
       f"mms{probe}_J'E_3",
       #f'mms{probe}_DE',
       ])



#var_label=[f'mms{probe}_mec_r_gse_re_x', f'mms{probe}_mec_r_gse_re_y', f'mms{probe}_mec_r_gse_re_z'], xsize=18, ysize=30)

#E'para = E'xBversor
#E'perp = sqrt(E'^2-E'par^2)
#E paralelo  = ExBversor
#E perp = sqrt(E^2-Epar^2)

#transformation_matrices = {
    #1: np.array([[0.03086242, -0.16024368, 0.9865949],
                 #[0.06529449, -0.9846335, -0.16196764],
                 #[0.99738866,  0.06941792, -0.01992514]]),
    #2: np.array([[0.0240478,  -0.13152222, 0.9910215],
                 #[0.0702289,  -0.98863685, -0.1329099],
                 #[0.997241,    0.07279454, -0.01453788]]),
    #3: np.array([[0.02380513, -0.14284547, 0.9894587],
                 #[0.06304695, -0.9875545,  -0.14408739],
                 #[0.9977266,   0.06581236, -0.01450289]]),
    #4: np.array([[0.01595144, -0.1127107,  0.9934998],
                 #[0.07349081, -0.9908065,  -0.11358511],
                 #[0.9971683,   0.07482495, -0.00752159]])