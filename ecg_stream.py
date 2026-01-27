import random

cycle = 0

def get_ecg_features():
    global cycle
    cycle += 1

    # -------------------------
    # HEALTHY (≈6 seconds)
    # -------------------------
    if cycle < 3:
        return {
            "RestingBP": random.randint(110, 125),
            "MaxHR": random.randint(130, 155),
            "ExerciseAngina": 0,
            "ChestPainType": "NAP",
            "RestingECG": "Normal",
            "Oldpeak": round(random.uniform(0.0, 0.6), 2),
            "ST_Slope": "Up",
            "Q Wave": 0,
            "Tinversion": 0,
            "LVH": 0,
            "EF-TTE": random.randint(60, 70),
            "Region RWMA": 0,
            "Dyspnea": 0,
            "Exertional CP": 0
        }

    # -------------------------
    # ISCHEMIA / WARNING (≈6 sec)
    # -------------------------
    elif cycle < 6:
        return {
            "RestingBP": random.randint(140, 165),
            "MaxHR": random.randint(100, 125),
            "ExerciseAngina": random.choice([0, 1]),
            "ChestPainType": random.choice(["NAP", "ASY"]),
            "RestingECG": "ST",
            "Oldpeak": round(random.uniform(1.2, 2.5), 2),
            "ST_Slope": "Flat",
            "Q Wave": random.choice([0, 1]),
            "Tinversion": random.choice([0, 1]),
            "LVH": random.choice([0, 1]),
            "EF-TTE": random.randint(45, 55),
            "Region RWMA": random.choice([0, 1]),
            "Dyspnea": random.choice([0, 1]),
            "Exertional CP": random.choice([0, 1])
        }

    # -------------------------
    # ARRHYTHMIA (≈6 sec)
    # -------------------------
    elif cycle < 9:
        return {
            "RestingBP": random.randint(150, 180),
            "MaxHR": random.randint(50, 90),
            "ExerciseAngina": 1,
            "ChestPainType": "ASY",
            "RestingECG": "LVH",
            "Oldpeak": round(random.uniform(2.5, 3.5), 2),
            "ST_Slope": "Down",
            "Q Wave": random.choice([0, 1]),
            "Tinversion": 1,
            "LVH": 1,
            "EF-TTE": random.randint(35, 45),
            "Region RWMA": 1,
            "Dyspnea": 1,
            "Exertional CP": 1
        }

    # -------------------------
    # ACUTE MI / SOS (≈6 sec)
    # -------------------------
    elif cycle < 12:
        return {
            "RestingBP": random.randint(170, 200),
            "MaxHR": random.randint(40, 70),
            "ExerciseAngina": 1,
            "ChestPainType": "ASY",
            "RestingECG": "MI",
            "Oldpeak": round(random.uniform(3.5, 5.0), 2),
            "ST_Slope": "Down",
            "Q Wave": 1,
            "Tinversion": 1,
            "LVH": 1,
            "EF-TTE": random.randint(25, 35),
            "Region RWMA": 1,
            "Dyspnea": 1,
            "Exertional CP": 1
        }

    # -------------------------
    # RESET
    # -------------------------
    else:
        cycle = 0
        return get_ecg_features()
