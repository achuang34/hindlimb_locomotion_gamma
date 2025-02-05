import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# Constants
M_PI = math.pi
Scale_Time = 0.55

# Muscle Index Mapping (9 entries, but will use 7 columns for muscle length and velocity)
M_IN = {
    "IP_H": 0, "GM_H": 1, "VL_K": 2, "TA_A": 3, "SO_A": 4, 
    "BF_H": 5, "BF_K": 6, "GA_K": 7, "GA_A": 8
}

# Angle-to-Length Mapping (9 entries corresponding to each joint-muscle pair)
angle2length = [0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.0025, 0.0075]

# Neutral Angles (in Degrees)
neutral_angles = {"hip": 65.0, "knee": -90.0, "ankle": 100.0}
neutral_angles_rad = {k: v * (M_PI / 180.0) for k, v in neutral_angles.items()}

# Load Excel file
df = pd.read_excel("/Users/austinchuang/documents/github/hindlimb-locomotion/python/out/model2.1.xlsx")

# Extract time and relevant columns
time = df['time']
angles = df[['T0', 'T11', 'T12', 'T13', 'T21', 'T22', 'T23']].values
velocities = df[['DT0', 'DT11', 'DT12', 'DT13', 'DT21', 'DT22', 'DT23']].values


# Create Muscle Length and Velocity arrays with 7 columns
MuscleLength = np.zeros((len(time), 7))
MuscleVelocity = np.zeros((len(time), 7))

# Compute muscle length and velocity for each muscle
# IP (Hip flexion)
idx = M_IN["IP_H"]
neutral_correction = neutral_angles_rad["hip"]
MuscleLength[:, 0] = 0.85 * (1.0 - angle2length[idx] * (angles[:, 1] - neutral_correction) / (M_PI / 180.0))
MuscleVelocity[:, 0] = -0.85 * angle2length[idx] * velocities[:, 1] / (M_PI / 180.0) * Scale_Time

# GM (Hip extension)
idx = M_IN["GM_H"]
neutral_correction = neutral_angles_rad["hip"]
MuscleLength[:, 1] = 0.85 * (1.0 + angle2length[idx] * (angles[:, 1] - neutral_correction) / (M_PI / 180.0))
MuscleVelocity[:, 1] = 0.85 * angle2length[idx] * velocities[:, 1] / (M_PI / 180.0) * Scale_Time

# VL (Knee extension)
idx = M_IN["VL_K"]
neutral_correction = neutral_angles_rad["knee"]
MuscleLength[:, 2] = 0.85 * (1.0 - angle2length[idx] * (angles[:, 2] - neutral_correction) / (M_PI / 180.0))
MuscleVelocity[:, 2] = -0.85 * angle2length[idx] * velocities[:, 2] / (M_PI / 180.0) * Scale_Time

# TA (Ankle flexion)
idx = M_IN["TA_A"]
neutral_correction = neutral_angles_rad["ankle"]
MuscleLength[:, 3] = 0.85 * (1.0 - angle2length[idx] * (angles[:, 3] - neutral_correction) / (M_PI / 180.0))
MuscleVelocity[:, 3] = -0.85 * angle2length[idx] * velocities[:, 3] / (M_PI / 180.0) * Scale_Time

# SO (Ankle extension)
idx = M_IN["SO_A"]
neutral_correction = neutral_angles_rad["ankle"]
MuscleLength[:, 4] = 0.85 * (1.0 + angle2length[idx] * (angles[:, 3] - neutral_correction) / (M_PI / 180.0))
MuscleVelocity[:, 4] = 0.85 * angle2length[idx] * velocities[:, 3] / (M_PI / 180.0) * Scale_Time

# BF (Hip extension + Knee flexion)
idx_hip = M_IN["BF_H"]
idx_knee = M_IN["BF_K"]
MuscleLength[:, 5] = 0.75 * (1.0 + angle2length[idx_hip] * (angles[:, 1] - neutral_angles_rad["hip"]) / (M_PI / 180.0) +
                            angle2length[idx_knee] * (angles[:, 2] - neutral_angles_rad["knee"]) / (M_PI / 180.0))
MuscleVelocity[:, 5] = 0.75 * (angle2length[idx_hip] * velocities[:, 1] / (M_PI / 180.0) + 
                               angle2length[idx_knee] * velocities[:, 2] / (M_PI / 180.0)) * Scale_Time

# GA (Knee flexion + Ankle extension)
idx_knee = M_IN["GA_K"]
idx_ankle = M_IN["GA_A"]
MuscleLength[:, 6] = 0.75 * (1.0 + angle2length[idx_knee] * (angles[:, 2] - neutral_angles_rad["knee"]) / (M_PI / 180.0) +
                            angle2length[idx_ankle] * (angles[:, 3] - neutral_angles_rad["ankle"]) / (M_PI / 180.0))
MuscleVelocity[:, 6] = 0.75 * (angle2length[idx_knee] * velocities[:, 2] / (M_PI / 180.0) + 
                               angle2length[idx_ankle] * velocities[:, 3] / (M_PI / 180.0)) * Scale_Time


# Compute Ia afferent activity (example code from earlier)
def compute_Ia_afferent(muscle_velocity, muscle_length, input_signal, dthr, kv, kdI, kaI):
    num_timesteps, num_muscles = muscle_velocity.shape
    Ia_afferent = np.zeros((num_timesteps, num_muscles))
    for t in range(num_timesteps):
        for i in range(num_muscles):
            dnorm = max((muscle_length[t, i] - dthr) / (1.0 - dthr), 0)
            Ia_afferent[t, i] = ((max(muscle_velocity[t, i], 0) / dthr) ** 0.6 * kv + dnorm * kdI + input_signal[t, i] * kaI)
    return Ia_afferent

# Compute Type II afferent activity
def compute_II_afferent(muscle_length, input_signal, dthr, kdII, kaII):
    num_timesteps, num_muscles = muscle_length.shape
    II_afferent = np.zeros((num_timesteps, num_muscles))
    for t in range(num_timesteps):
        for i in range(num_muscles):
            dnorm = max((muscle_length[t, i] - dthr) / (1.0 - dthr), 0)  # Normalize length
            II_afferent[t, i] = dnorm * kdII + input_signal[t, i] * kaII
    return II_afferent# Compute Type II afferent activity

# Define afferent parameters
dthr = 0.85
kv, kdI, kdII, kaI, KaII = 6.2, 2.0, 1.5, 0.06, 0.06

file_path = "/Users/austinchuang/documents/github/hindlimb-locomotion/python/out/mnact2.1.xlsx"
isdf = pd.read_excel(file_path)
input_signal = isdf[['MnIL_L', 'MnST_L', 'MnVL_L', 'MnTA_L', 'MnSO_L', 'MnBF_L', 'MnGA_L']].values
#print(input_signal)
Ia_calculated = compute_Ia_afferent(MuscleVelocity, MuscleLength, input_signal, dthr, kv, kdI, kaI)
II_calculated = compute_II_afferent(MuscleLength, input_signal, dthr, kdII, KaII)

# Load the data from the Excel file with "real" Ia afferent activity
file_path2 = "/Users/austinchuang/documents/github/hindlimb-locomotion/python/out/Ia_simdata2.1.xlsx"
df_real = pd.read_excel(file_path2)

# Extract time and relevant columns for real Ia afferent data
Ia_columns = ['fb_Ia_IL_L', 'fb_Ia_ST_L', 'fb_Ia_VL_L', 'fb_Ia_TA_L', 'fb_Ia_SO_L', 'fb_Ia_BF_L', 'fb_Ia_GA_L']
Ia_data = df_real[Ia_columns]

# Load the data from the Excel file with "real" Ia afferent activity
file_path3 = "/Users/austinchuang/documents/github/hindlimb-locomotion/python/out/Ia_simdata2.1.xlsx"
df_real = pd.read_excel(file_path3)

# Extract time and relevant columns for real Ia afferent data
II_columns = ['fb_II_IL_L', 'fb_II_ST_L', 'fb_II_VL_L', 'fb_II_TA_L', 'fb_II_SO_L', 'fb_II_BF_L', 'fb_II_GA_L']
II_data = df_real[II_columns]


#switched fb_Ia_BF_L and fb_Ia_ST_L, so index 1 is BF and index 5 is ST/GM
plt.figure(figsize=(12, 8))
plt.plot(time, II_calculated[:, 0], label = 'II_calculated')
plt.plot(time, II_data.values[:, 0], label = 'II_data.values')
plt.xlabel('Time')
plt.ylabel('Ia Afferent Activity')
plt.title('IP')
plt.legend()
plt.show()





# Compute the difference between real and calculated Ia afferent activity
# Ia_diff = Ia_data.values - Ia_calculated
# print(Ia_data.values)
# print(Ia_calculated)
# print(Ia_diff)

#     # Plot the difference in Ia afferent activity
# plt.figure(figsize=(12, 8))
# for i, col in enumerate(Ia_columns):
#         plt.plot(time, Ia_diff[:, i], label=f'{col} Difference')

# plt.xlabel('Time')
# plt.ylabel('Ia Afferent Activity Difference')
# plt.title('Difference Between Real and Calculated Ia Afferent Activity')
# plt.legend()
# plt.grid(True)
# plt.show()
