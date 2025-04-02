import pandas as pd
import numpy as np

data = np.load("full_patient_details.npy", allow_pickle=True)
df = pd.DataFrame(data, columns=["Filename", "Sex", "Age", "Size (m)", "Weight (kg)","row","column"])
print(df.head(20))
