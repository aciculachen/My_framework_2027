"""Facts about the recording setup: marker count, sample rates, carrier, radar position, skeleton."""
from typing import Dict, List

import numpy as np

# --- Data Constants ---
N_MARKER = 53

# --- Physical Constants ---
C_MPS = 299_792_458.0  # Speed of light [m/s]

# --- Hardware & Setup Constants ---
RD_SP_HZ = 256  # Radar sampling rate [Hz]
MC_SP_HZ = 240  # Mocap sampling rate [Hz]
RADAR_FC_HZ = 5.8e9  # Carrier frequency [Hz]
RADAR_WAVELENGTH = C_MPS / RADAR_FC_HZ  # Radar wavelength [m]

# Radar positions [mm]
_RADAR_POS_MM = np.array([
    [-32.621090, -3601.985107, 734.453247],
    [-172.841934, -3567.285156, 676.106445],
    [-69.292000, -3563.932129, 674.375427],
    [-148.522736, -3608.971924, 742.861145],
])
# Export processed values in SI units [m]
RADAR_POS_M = _RADAR_POS_MM * 1e-3

# -----------------------------------------------------------------
# Mocap Skeleton Definition
# -----------------------------------------------------------------

MOCAP_NODES: List[str] = [
'ARIEL', 'LFHD', 'LBHD', 'RFHD', 'RBHD', 'C7', 'T10', 'CLAV', 'STRN', 'LFSH',
'LBSH', 'LUPA', 'LELB', 'LIEL', 'LFRM', 'LIWR', 'LOWR', 'LIHAND', 'LOHAND', 'RFSH',
'RBSH', 'RUPA', 'RELB', 'RIEL', 'RFRM', 'RIWR', 'ROWR', 'RIHAND', 'ROHAND', 'LFWT',
'MFWT', 'RFWT', 'LBWT', 'MBWT', 'RBWT', 'LTHI', 'LKNE', 'LKNI', 'LSHN', 'LANK',
'LHEL', 'LMT5', 'LMT1', 'LTOE', 'RTHI', 'RKNE', 'RKNI', 'RSHN', 'RANK', 'RHEL',
'RMT5', 'RMT1', 'RTOE'
]


# Segment -> markers mapping (based on common Plug-in Gait conventions and provided labels)
MOCAP_SEGMENTS: Dict[str, List[str]] = {
# Head / neck / torso
"Head": ["ARIEL", "LFHD", "RFHD", "LBHD", "RBHD"],
"Torso": ["C7", "T10", "CLAV", "STRN"],

# Shoulders
"LeftShoulder": ["LFSH", "LBSH"],
"RightShoulder": ["RFSH", "RBSH"],

# Upper arms (include elbow landmarks to show segment span)
"LeftUpperArm": ["LUPA", "LELB", "LIEL"],
"RightUpperArm": ["RUPA", "RELB", "RIEL"],

# Forearms (include wrist landmarks)
"LeftForearm": ["LFRM", "LIWR", "LOWR"],
"RightForearm": ["RFRM", "RIWR", "ROWR"],

# Hands
"LeftHand": ["LIHAND", "LOHAND"],
"RightHand": ["RIHAND", "ROHAND"],

# Pelvis (front/back, left/right, and mid points as available)
"Pelvis": ["LFWT", "MFWT", "RFWT", "LBWT", "MBWT", "RBWT"],

# Thighs (include knee landmarks)
"LeftThigh": ["LTHI", "LKNE", "LKNI"],
"RightThigh": ["RTHI", "RKNE", "RKNI"],

# Shanks (include ankle)
"LeftShank": ["LSHN", "LANK"],
"RightShank": ["RSHN", "RANK"],

# Feet (heel, metatarsals, toe)
"LeftFoot": ["LHEL", "LMT5", "LMT1", "LTOE"],
"RightFoot": ["RHEL", "RMT5", "RMT1", "RTOE"],
}