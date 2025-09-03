#módulos
import pyspedas
from pytplot import get_data
from pytplot.tplot_math import time_clip as tclip
import numpy as np
from pyspedas import time_double
import matplotlib.pyplot as plt

#----------------------------------------------------------------------------------------------------

# def - time
#time MP, reg1 e reg2+
start_timeMP = '2024/3/26 17:12:00'
end_timeMP  = '2024/3/26 17:12:05'
trangeMP = [start_timeMP, end_timeMP]
probe = ['1']

var11 = float(input("Início Sheath:"))
var12 = float(input("Fim Sheath:"))
start_timeSH = time_double(start_timeMP) + (var11)
end_timeSH   = time_double(start_timeMP) + (var12)
trangeSH = [start_timeSH, end_timeSH]
probe = ['1']

var21= float(input ("Início Sphere:"))
var22= float(input ("Fim Sphere:"))
start_timeSP = time_double(start_timeMP) + (var21)
end_timeSP = time_double(start_timeMP) + (var22)
trangeSP = [start_timeSP, end_timeSP]
probe = ['1']

if time_double(start_timeSH) >= time_double(end_timeSP) :
    total_trange = [start_timeSP, end_timeSH]
else:
     total_trange = [start_timeSH, end_timeSP]

#data 
pyspedas.mms.fgm(probe=probe, trange=total_trange, time_clip=True)
#dados FGM (Bmag,Bx,By,Bz)
pyspedas.mms.fpi(probe=probe, trange=total_trange, datatype='dis-moms', data_rate= 'brst',time_clip=True)
#dados FPI (íon flux, density, velocity)
#----------------------------------------------------------------------------------------------------

B1 ='mms1_fgm_b_gsm_srvy_l2_btot'
tclip(B1, time_start = start_timeSH, time_end = end_timeSH, suffix='sHeath')
t_sHeath, B1_sHeath = get_data(B1+'sHeath')
av_BSH = np.nanmean(B1_sHeath, axis = 0)

B2 ='mms1_fgm_b_gsm_srvy_l2_btot'
tclip(B2, time_start = start_timeSP, time_end = end_timeSP, suffix='sPhere')
t_sPhere, B2_sPhere = get_data(B2+'sPhere')
av_BSP = np.nanmean(B2_sPhere, axis = 0)

B3 ='mms1_fgm_b_gsm_srvy_l2_bvec'
tclip(B3, time_start = start_timeSH, time_end = end_timeSH, suffix='sHeath')
t_sHeath, B3_sHeath = get_data(B3+'sHeath')
av_BvecSH = np.nanmean(B3_sHeath, axis = 0)

B4 ='mms1_fgm_b_gsm_srvy_l2_bvec'
tclip(B4, time_start = start_timeSP, time_end = end_timeSP, suffix='sPhere')
t_sPhere, B4_sPhere = get_data(B4+'sPhere')
av_BvecSP = np.nanmean(B4_sPhere, axis = 0)

N1 = 'mms1_dis_numberdensity_brst'
tclip(N1, time_start = start_timeSH, time_end = end_timeSH, suffix='sHeath')
t_sHeath, N1_sHeath = get_data(N1+'sHeath')
av_NSH = np.nanmean(N1_sHeath, axis = 0)

N2 = 'mms1_dis_numberdensity_brst'
tclip(N2, time_start = start_timeSP, time_end = end_timeSP, suffix='sPhere')
t_sPhere, N2_sPhere = get_data(N2+'sPhere')
av_NSP = np.nanmean(N2_sPhere, axis = 0)

Tpp1 = 'mms1_dis_tempperp_brst'
tclip(Tpp1, time_start = start_timeSH, time_end = end_timeSH, suffix='sHeath')
t_sHeath, Tpp_sHeath = get_data(Tpp1+'sHeath')
av_TppSH = np.nanmean(Tpp_sHeath, axis = 0)

Tpp2 = 'mms1_dis_tempperp_brst'
tclip(Tpp2, time_start = start_timeSP, time_end = end_timeSP, suffix='sPhere')
t_sPhere, Tpp_sPhere = get_data(Tpp2+'sPhere')
av_TppSP = np.nanmean(Tpp_sPhere, axis = 0)

Tpa1 = 'mms1_dis_temppara_brst'
tclip(Tpa1, time_start = start_timeSH, time_end = end_timeSH, suffix='sHeath')
t_sHeath, Tpa_sHeath = get_data(Tpa1+'sHeath')
av_TpaSH = np.nanmean(Tpa_sHeath, axis = 0)

Tpa2 = 'mms1_dis_temppara_brst'
tclip(Tpa2, time_start = start_timeSP, time_end = end_timeSP, suffix='sPhere')
t_sPhere, Tpa_sPhere = get_data(Tpa2+'sPhere')
av_TpaSP= np.nanmean(Tpa_sPhere, axis = 0)

mi0 = 4*np.pi*10e-7
#B1= 10sSheath, B2= 10sSphere
#----------------------------------------------------------------------------------------------------
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
