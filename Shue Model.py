import numpy as np

xgsm = float(input("X GSM:"))
ygsm = float(input("Y GSM:"))
zgsm = float(input("Z GSM:"))
Rgsm = np.sqrt(xgsm**2 + ygsm**2 + zgsm**2)
print("Rsat", Rgsm)

Rarc = np.sqrt((ygsm)**(2) + (zgsm)**(2))
theta = (np.arctan(Rarc/xgsm))
thetaDG = np.degrees(theta)

av_bz_sub = float(input("Bz:"))
av_p_sub = float(input("Dp:"))

#Aplicar Shue - Posição
r0 = (10.22 + 1.29 * np.tanh(0.184 * (av_bz_sub + 8.14))) / (av_p_sub**(1/6.6))
alpha = (0.58 - 7e-3 * av_bz_sub) * (1 + 24e-3 * np.log(av_p_sub))
r = r0 * (2/(1 + np.cos(theta)))**alpha

print ("R0:", r0)
print ("alpha:", alpha)
print ("theta:", thetaDG)
print ("R:", r)