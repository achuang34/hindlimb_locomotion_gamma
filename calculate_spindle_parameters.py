import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# Constants
M_PI = math.pi
Scale_Time = 0.35

# Muscle Index Mapping (9 entries, but will use 7 columns for muscle length and velocity)
M_IN = {
    "IP_H": 0, "GM_H": 1, "VL_K": 2, "TA_A": 3, "SO_A": 4, 
    "BF_H": 5, "BF_K": 6, "GA_K": 7, "GA_A": 8
}

# Angle-to-Length Mapping (9 entries corresponding to each joint-muscle pair)
angle2length = [0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.0025, 0.0075]

# Neutral Angles (in Degrees)
neutral_angles = {"hip": 50.0, "knee": -60.0, "ankle": 65.0}
neutral_angles_rad = {k: v * (M_PI / 180.0) for k, v in neutral_angles.items()}

# Load Excel file
df = pd.read_excel("./python/out/model2.1.xlsx")

# Extract time and relevant columns
time = df['time']
angles = df[['T0', 'T11', 'T12', 'T13', 'T21', 'T22', 'T23']].values
velocities = df[['DT0', 'DT11', 'DT12', 'DT13', 'DT21', 'DT22', 'DT23']].values

#neutral correction for each joint
#hip
neutral_correction = np.pi/2.0 - neutral_angles_rad["hip"]                  
neutral_correction_hip = (angles[:, 1] - neutral_correction + np.pi)%(np.pi*2.) - np.pi  

#knee
neutral_correction = - np.pi - neutral_angles_rad["knee"]                  
neutral_correction_knee = (angles[:, 2] - neutral_correction + np.pi)%(np.pi*2.) - np.pi 

#ankle
neutral_correction = np.pi - neutral_angles_rad["ankle"]                  
neutral_correction_ankle = (angles[:, 3] - neutral_correction + np.pi)%(np.pi*2.) - np.pi 

# Create Muscle Length and Velocity arrays with 7 columns
MuscleLength = np.zeros((len(time), 7))
MuscleVelocity = np.zeros((len(time), 7))

# Compute muscle length and velocity for each muscle
# IP (Hip flexion)
idx = M_IN["IP_H"]
MuscleLength[:, 0] = 0.85 * (1.0 - angle2length[idx] * (neutral_correction_hip) / (M_PI / 180.0))
MuscleVelocity[:, 0] = -0.85 * angle2length[idx] * velocities[:, 1] / (M_PI / 180.0) * Scale_Time

# GM (Hip extension)
idx = M_IN["GM_H"]
MuscleLength[:, 1] = 0.85 * (1.0 + angle2length[idx] * (neutral_correction_hip) / (M_PI / 180.0))
MuscleVelocity[:, 1] = 0.85 * angle2length[idx] * velocities[:, 1] / (M_PI / 180.0) * Scale_Time

# VL (Knee extension)
idx = M_IN["VL_K"]
MuscleLength[:, 2] = 0.85 * (1.0 - angle2length[idx] * (neutral_correction_knee) / (M_PI / 180.0))
MuscleVelocity[:, 2] = -0.85 * angle2length[idx] * velocities[:, 2] / (M_PI / 180.0) * Scale_Time

# TA (Ankle flexion)
idx = M_IN["TA_A"]
MuscleLength[:, 3] = 0.85 * (1.0 - angle2length[idx] * (neutral_correction_ankle) / (M_PI / 180.0))
MuscleVelocity[:, 3] = -0.85 * angle2length[idx] * velocities[:, 3] / (M_PI / 180.0) * Scale_Time

# SO (Ankle extension)
idx = M_IN["SO_A"]
MuscleLength[:, 4] = 0.85 * (1.0 + angle2length[idx] * (neutral_correction_ankle) / (M_PI / 180.0))
MuscleVelocity[:, 4] = 0.85 * angle2length[idx] * velocities[:, 3] / (M_PI / 180.0) * Scale_Time

# BF (Hip extension + Knee flexion)
idx_hip = M_IN["BF_H"]
idx_knee = M_IN["BF_K"]
MuscleLength[:, 5] = 0.75 * (1.0 + angle2length[idx_hip] * (neutral_correction_hip) / (M_PI / 180.0) +
                            angle2length[idx_knee] * (neutral_correction_knee) / (M_PI / 180.0))
MuscleVelocity[:, 5] = 0.75 * (angle2length[idx_hip] * velocities[:, 1] / (M_PI / 180.0) + 
                               angle2length[idx_knee] * velocities[:, 2] / (M_PI / 180.0)) * Scale_Time

# GA (Knee flexion + Ankle extension)
idx_knee = M_IN["GA_K"]
idx_ankle = M_IN["GA_A"]
MuscleLength[:, 6] = 0.75 * (1.0 + angle2length[idx_knee] * (neutral_correction_knee) / (M_PI / 180.0) +
                            angle2length[idx_ankle] * (neutral_correction_ankle) / (M_PI / 180.0))
MuscleVelocity[:, 6] = 0.75 * (angle2length[idx_knee] * velocities[:, 2] / (M_PI / 180.0) + 
                               angle2length[idx_ankle] * velocities[:, 3] / (M_PI / 180.0)) * Scale_Time


#Define afferent parameters
dthr = 0.85
kv, kdI, kdII, kaI, KaII = 6.2, 2.0, 1.5, 0.06, 0.06
dnorm = np.maximum((MuscleLength - dthr) / (1.0 - dthr), 0)

#Load motor neuron activation inputs
file_path = "./python/out/mnact2.1.xlsx"
isdf = pd.read_excel(file_path)
input_signal = isdf[['MnIL_L', 'MnBF_L', 'MnVL_L', 'MnTA_L', 'MnSO_L', 'MnST_L' , 'MnGA_L']].values

#Calculate afferents
II_calculated = dnorm * kdII + input_signal * KaII
Ia_calculated = ((np.maximum(MuscleVelocity, 0) / dthr) ** 0.6 * kv + dnorm * kdI + input_signal * kaI)

# Load the data from the Excel file with "real" Ia afferent activity
file_path2 = "./python/out/Ia_simdata2.1.xlsx"
df_real = pd.read_excel(file_path2)

# Extract time and relevant columns for real Ia afferent data
Ia_columns = ['fb_Ia_IL_L','fb_Ia_BF_L' , 'fb_Ia_VL_L', 'fb_Ia_TA_L', 'fb_Ia_SO_L', 'fb_Ia_ST_L', 'fb_Ia_GA_L']
Ia_data = df_real[Ia_columns]

# Load the data from the Excel file with "real" Ia afferent activity
file_path3 = "./python/out/Ia_simdata2.1.xlsx"
df_real = pd.read_excel(file_path3)

# Extract time and relevant columns for real Ia afferent data
II_columns = ['fb_II_IL_L', 'fb_II_BF_L' , 'fb_II_VL_L', 'fb_II_TA_L', 'fb_II_SO_L', 'fb_II_ST_L', 'fb_II_GA_L']
II_data = df_real[II_columns]


#Save data of velocity, length, and afferents to Excel file
#Create rows of min/max values for each muscle
muscles_list = ['IL', 'BF', 'VL', 'TA', 'SO', 'ST', 'GA']
length_maximums = np.max(MuscleLength, axis=0)
length_minimums = np.min(MuscleLength, axis=0)
velocity_maximums = np.max(MuscleVelocity, axis=0)
velocity_minimums = np.min(MuscleVelocity, axis=0)
II_maximums = np.max(II_calculated, axis=0)
II_minimums = np.min(II_calculated, axis=0)
Ia_maximums = np.max(Ia_calculated, axis=0)
Ia_minimums = np.min(Ia_calculated, axis=0)

# Reshape the 1D arrays into 2D (1 row per variable, 7 columns for muscles)
length_maximums = length_maximums.reshape(1, -1)
length_minimums = length_minimums.reshape(1, -1)
velocity_maximums = velocity_maximums.reshape(1, -1)
velocity_minimums = velocity_minimums.reshape(1, -1)
II_maximums = II_maximums.reshape(1, -1)
II_minimums = II_minimums.reshape(1, -1)
Ia_maximums = Ia_maximums.reshape(1, -1)
Ia_minimums = Ia_minimums.reshape(1, -1)

#Create dataframes for each muscle
df_length_maximums = pd.DataFrame(data=length_maximums, columns=muscles_list)
df_length_minimums = pd.DataFrame(data=length_minimums, columns=muscles_list)
df_velocity_maximums = pd.DataFrame(data=velocity_maximums, columns=muscles_list)
df_velocity_minimums = pd.DataFrame(data=velocity_minimums, columns=muscles_list)
df_II_maximums = pd.DataFrame(data=II_maximums, columns=muscles_list)
df_II_minimums = pd.DataFrame(data=II_minimums, columns=muscles_list)
df_Ia_maximums = pd.DataFrame(data=Ia_maximums, columns=muscles_list)
df_Ia_minimums = pd.DataFrame(data=Ia_minimums, columns=muscles_list)

#Insert label column for each dataframe
df_length_maximums.insert(0, 'Label', 'Length Maximums')
df_length_minimums.insert(0, 'Label', 'Length Minimums')
df_velocity_maximums.insert(0, 'Label', 'Velocity Maximums')
df_velocity_minimums.insert(0, 'Label', 'Velocity Minimums')
df_II_maximums.insert(0, 'Label', 'II Maximums')
df_II_minimums.insert(0, 'Label', 'II Minimums')
df_Ia_maximums.insert(0, 'Label', 'Ia Maximums')
df_Ia_minimums.insert(0, 'Label', 'Ia Minimums')

#Combine dataframes
df_combined = pd.concat([df_length_maximums, df_length_minimums, df_velocity_maximums, 
                         df_velocity_minimums, df_II_maximums, df_II_minimums, df_Ia_maximums, 
                         df_Ia_minimums])
#Save to Excel file
#df_combined.to_excel('./out/w_parameters.xlsx', index=False)



#switched fb_Ia_BF_L and fb_Ia_ST_L, so index 1 is BF and index 5 is ST/GM
#Plot afferent activities
# plt.figure(figsize=(12, 8))
# plt.plot(time, II_calculated[:, 5], label = 'II_calculated')
# plt.plot(time, II_data.values[:, 5], label = 'II_data.values')
# plt.plot(time, Ia_calculated[:, 5], label = 'Ia_calculated')
# plt.plot(time, Ia_data.values[:, 5], label = 'Ia_data.values')
# plt.xlabel('Time')
# plt.ylabel('Ia Afferent Activity')
# plt.title('IP')
# plt.legend()
# plt.show()

#plot muscle length and velocity
plt.figure(figsize=(12, 8))
plt.plot(time, MuscleLength[:, 0]/MuscleVelocity, label = ['IP'])
#plt.plot(time, MuscleVelocity[:, 5], label = 'Muscle Velocity')
plt.xlabel('Time')
plt.ylabel('Muscle Length')
plt.title('Muscle Length')
plt.legend()
plt.show()

