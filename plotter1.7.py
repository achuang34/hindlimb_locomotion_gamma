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

# Angle-to-Length Mapping (9 entries corresponding to each joint)
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

# Now `MuscleLength` and `MuscleVelocity` have 7 columns for the 7 muscles

# Compute Ia afferent activity 
def compute_Ia_afferent(muscle_velocity, muscle_length, input_signal, dthr, kv, kdI, kaI):
    num_timesteps, num_muscles = muscle_velocity.shape
    Ia_afferent = np.zeros((num_timesteps, num_muscles))
    for t in range(num_timesteps):
        for i in range(num_muscles):
            dnorm = max((muscle_length[t, i] - dthr) / (1.0 - dthr), 0)
            Ia_afferent[t, i] = (max(muscle_velocity[t, i], 0) / dthr) ** 0.6 * kv + dnorm * kdI + input_signal[t, i] * kaI
    return Ia_afferent

# Define Ia afferent parameters
dthr = 0.1
kv, kdI, kaI = 1.5, 0.8, 0.3
input_signal = np.random.rand(len(time), 7)
Ia_calculated = compute_Ia_afferent(MuscleVelocity, MuscleLength, input_signal, dthr, kv, kdI, kaI)

print("Ia Afferent Activity:\n", Ia_calculated)

# Plot Muscle Length, Muscle Velocity, and Ia Afferent Activity vs Time in a Single Plot
fig, axs = plt.subplots(3, 1, figsize=(10, 8))

# Muscle Length vs Time
for i in range(7):
    axs[0].plot(time, MuscleLength[:, i], label=f'Muscle {i+1}')
axs[0].set_title('Muscle Length vs Time')
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Muscle Length (m)')
axs[0].legend(loc='upper right')
axs[0].grid(True)

# Muscle Velocity vs Time
for i in range(7):
    axs[1].plot(time, MuscleVelocity[:, i], label=f'Muscle {i+1}')
axs[1].set_title('Muscle Velocity vs Time')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Muscle Velocity (m/s)')
axs[1].legend(loc='upper right')
axs[1].grid(True)

# Ia Afferent Activity vs Time
for i in range(7):
    axs[2].plot(time, Ia_calculated[:, i], label=f'Muscle {i+1}')
axs[2].set_title('Ia Afferent Activity vs Time')
axs[2].set_xlabel('Time (s)')
axs[2].set_ylabel('Ia Afferent Activity')
axs[2].legend(loc='upper right')
axs[2].grid(True)

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()

# # Plot results individually
# # Plot Muscle Length vs Time
# plt.figure(figsize=(12, 6))
# for i in range(7):
#     plt.plot(time, MuscleLength[:, i], label=f'Muscle {i+1}')
# plt.title('Muscle Length vs Time')
# plt.xlabel('Time (s)')
# plt.ylabel('Muscle Length (m)')
# plt.legend(loc='upper right')
# plt.grid(True)
# plt.show()

# # Plot Muscle Velocity vs Time
# plt.figure(figsize=(12, 6))
# for i in range(7):
#     plt.plot(time, MuscleVelocity[:, i], label=f'Muscle {i+1}')
# plt.title('Muscle Velocity vs Time')
# plt.xlabel('Time (s)')
# plt.ylabel('Muscle Velocity (m/s)')
# plt.legend(loc='upper right')
# plt.grid(True)
# plt.show()

# # Plot Ia Afferent Activity vs Time
# plt.figure(figsize=(12, 6))
# for i in range(7):
#     plt.plot(time, Ia_calculated[:, i], label=f'Muscle {i+1}')
# plt.title('Ia Afferent Activity vs Time')
# plt.xlabel('Time (s)')
# plt.ylabel('Ia Afferent Activity')
# plt.legend(loc='upper right')
# plt.grid(True)
# plt.show()
