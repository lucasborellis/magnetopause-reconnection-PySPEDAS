#módulos
import pyspedas
import pandas as pd
from pytplot import get_data, store_data
from pytplot.tplot_math import time_clip as tclip
import numpy as np
from pyspedas import time_double
import matplotlib.pyplot as plt

#----------------------------------------------------------------------------------------------------
#Velocidade de outlfow obtida para o evento:
Voutflow = np.array([144.86917807, -74.08841113 , 29.50946557])

# def - time
#time MP, reg1 e reg2+
start_timeMP = '2024/3/26 17:12:00'
end_timeMP  = '2024/3/26 17:12:30'
trangeMP = [start_timeMP, end_timeMP]
probe = 1

var11 = float(input("Início Sheath:"))
var12 = float(input("Fim Sheath:"))
start_timeSH = time_double(start_timeMP) + (var11)
end_timeSH   = time_double(start_timeMP) + (var12)
trangeSH = [start_timeSH, end_timeSH]

var21= float(input ("Início Sphere:"))
var22= float(input ("Fim Sphere:"))
start_timeSP = time_double(start_timeMP) + (var21)
end_timeSP = time_double(start_timeMP) + (var22)
trangeSP = [start_timeSP, end_timeSP]

if time_double(start_timeSH) >= time_double(end_timeSP) :
    total_trange = [start_timeSP, end_timeSH]
else:
     total_trange = [start_timeSH, end_timeSP]

pyspedas.mms.fgm(probe=probe, trange=total_trange, data_rate='brst', time_clip=True)
pyspedas.mms.fpi(probe=probe, trange=total_trange, data_rate='brst', datatype='dis-moms', 
                 varnames=[f'mms{probe}_dis_numberdensity_brst', f'mms{probe}_dis_tempperp_brst',
                           f'mms{probe}_dis_temppara_brst'], time_clip=True)
#----------------------------------------------------------------------------------------------------
#1- Walen Test

mi0 = 4*np.pi*10e-7 
#B1= 10sSheath, B2= 10sSphere

B1 =f'mms{probe}_fgm_b_gsm_brst_l2_btot'
B2 =f'mms{probe}_fgm_b_gsm_brst_l2_btot'
B3 =f'mms{probe}_fgm_b_gsm_brst_l2_bvec'
B4 =f'mms{probe}_fgm_b_gsm_brst_l2_bvec'

transformation_matrix = np.array([[0.03086242, -0.16024368, 0.9865949],
                 [0.06529449, -0.9846335, -0.16196764],
                 [0.99738866,  0.06941792, -0.01992514]])#(vetores L, M, N como linhas)

timeB, B = get_data(f'mms{probe}_fgm_b_gsm_brst_l2') 
B_gsm = B[:, 0:3]
B_lmn = np.dot(B_gsm, transformation_matrix.T) # Realiza a transformação para LMN
store_data(f'mms{probe}_fgm_b_lmn_brst_l2_vec', data={'x': timeB, 'y': B_lmn})

B_total = np.sqrt(np.sum(B_lmn**2, axis=1))  # Calcula o módulo ao longo das linhas
#B_combined = np.hstack((B_lmn, B_total[:, None]))  # Adiciona B_total como a 4ª coluna
store_data(f'mms{probe}_fgm_b_lmn_brst_l2_tot', data={'x': timeB, 'y': B_total})

B1=B2= f'mms{probe}_fgm_b_lmn_brst_l2_tot'
B3=B4= f'mms{probe}_fgm_b_lmn_brst_l2_vec'
N1= f'mms{probe}_dis_numberdensity_brst'
N2= f'mms{probe}_dis_numberdensity_brst'
Tpp1 = f'mms{probe}_dis_tempperp_brst'
Tpp2 = f'mms{probe}_dis_tempperp_brst'
Tpa1 = f'mms{probe}_dis_temppara_brst'
Tpa2 = f'mms{probe}_dis_temppara_brst'

tclip(B1, time_start = start_timeSH, time_end = end_timeSH, suffix='sHeath')
t_sHeath, B1_sHeath = get_data(B1+'sHeath')
av_BSH = np.nanmean(B1_sHeath, axis = 0)
tclip(B2, time_start = start_timeSP, time_end = end_timeSP, suffix='sPhere')
t_sPhere, B2_sPhere = get_data(B2+'sPhere')
av_BSP = np.nanmean(B2_sPhere, axis = 0)
tclip(B3, time_start = start_timeSH, time_end = end_timeSH, suffix='sHeath')
t_sHeath, B3_sHeath = get_data(B3+'sHeath')
av_BvecSH = np.nanmean(B3_sHeath, axis = 0)
tclip(B4, time_start = start_timeSP, time_end = end_timeSP, suffix='sPhere')
t_sPhere, B4_sPhere = get_data(B4+'sPhere')
av_BvecSP = np.nanmean(B4_sPhere, axis = 0)
tclip(N1, time_start = start_timeSH, time_end = end_timeSH, suffix='sHeath')
t_sHeath, N1_sHeath = get_data(N1+'sHeath')
av_NSH = np.nanmean(N1_sHeath, axis = 0)
tclip(N2, time_start = start_timeSP, time_end = end_timeSP, suffix='sPhere')
t_sPhere, N2_sPhere = get_data(N2+'sPhere')
av_NSP = np.nanmean(N2_sPhere, axis = 0)
tclip(Tpp1, time_start = start_timeSH, time_end = end_timeSH, suffix='sHeath')
t_sHeath, Tpp_sHeath = get_data(Tpp1+'sHeath')
av_TppSH = np.nanmean(Tpp_sHeath, axis = 0)
tclip(Tpp2, time_start = start_timeSP, time_end = end_timeSP, suffix='sPhere')
t_sPhere, Tpp_sPhere = get_data(Tpp2+'sPhere')
av_TppSP = np.nanmean(Tpp_sPhere, axis = 0)
tclip(Tpa1, time_start = start_timeSH, time_end = end_timeSH, suffix='sHeath')
t_sHeath, Tpa_sHeath = get_data(Tpa1+'sHeath')
av_TpaSH = np.nanmean(Tpa_sHeath, axis = 0)
tclip(Tpa2, time_start = start_timeSP, time_end = end_timeSP, suffix='sPhere')
t_sPhere, Tpa_sPhere = get_data(Tpa2+'sPhere')
av_TpaSP= np.nanmean(Tpa_sPhere, axis = 0)

PpeSH  = av_NSH*av_TppSH*(1.602e-4)
PpaSH = av_NSH*av_TpaSH*(1.602e-4)
alphaSH = ((PpaSH- PpeSH)*((mi0)/av_BSH**2))

PpeSP  = av_NSP*av_TppSP *(1.602e-4)
PpaSP= av_NSP*av_TpaSP *(1.602e-4)
alphaSP = ((PpaSP - PpeSP)*((mi0)/av_BSP**2)) 

Vpred = 1e-12*(((1-alphaSH)**(1/2))*((mi0*av_NSH*1.672e-21)**(-1/2))*((av_BvecSP*((1-alphaSP)/(1-alphaSH))-av_BvecSH))) 
VpredN = np.linalg.norm(Vpred)

V= np.abs(np.dot(Voutflow,Vpred)/(VpredN)**2)

#B-ShearAngle dependence

#cálculo do beta
Pb1 = (av_BSH**2)/(2*mi0)
Pth1 = 8254.40*av_NSH*((2*(av_TppSH) + av_TpaSH)/3)
BetasHeath= Pth1/Pb1 #Ter/mag

Pb2 = (av_BSP**2)/(2*mi0)
Pth2 = 8254.40*av_NSP*((2*(av_TppSP) + av_TpaSP)/3) #(CTE)=protonmass/bolztmann=8254.40
BetasPhere= Pth2/Pb2

if time_double(end_timeSH) <= time_double(start_timeMP) :
    deltaB= np.abs(BetasHeath - BetasPhere)
else: 
    deltaB= np.abs(BetasPhere - BetasHeath)

v1 = np.linalg.norm(av_BvecSH)
v2 = np.linalg.norm(av_BvecSP)
theta_SA =np.abs(np.degrees(np.arccos((np.dot(av_BvecSH, av_BvecSP))/(v1*v2)))) 
#----------------------------------------------------------------------------------------------------
# Plotando os dados
print ('Vpred:', Vpred)
print ('Voutflow:', Voutflow)
print ("ΔV*:",V)
print('Theta:' , theta_SA)
print('deltaBeta:' , deltaB)
theta = np.linspace(0, 180, 180)
plt.xscale('log')
plt.ylabel('θ (degrees)')
plt.xlabel('Δβ (βsheath - βsphere)')
plt.ylim(0,180)
plt.xlim(0.01,100)
plt.plot( deltaB , theta_SA, 'o')
plt.plot(2*np.tan(theta*np.pi/180/2.),theta,color='k')
plt.plot(np.tan(theta*np.pi/180/2.),theta, linestyle='--',color='k')
plt.plot(4*np.tan(theta*np.pi/180/2.),theta, linestyle='--',color='k')
plt.show()