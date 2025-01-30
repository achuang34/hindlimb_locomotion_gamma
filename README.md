using the plotters require downloading the h5 file from the hindlimb-locomotion model and extracting the information into excel files
Run simulation and press o twice, it will save the data from the time in between pressing both O's
this may require you to create an "out" and "recordings" folder in the python folder from hindlimb-locomotion

to extract as excel file:
bash
Pip install pandas
Pip install openpyxl
Python3

python
Import pandas as pd
# set file path
file_name = "file_path"

# df_fb is where Ia activity is stored in H5 file
df = pd.read_hdf(file_path, key = 'df_fb')
df.to_excel("file_path/Ia_simdata2.1.xlsx", index = False)

# df_model is where state variables are stored in H5 file
df = pd.read_hdf(file_path, key = 'df_model')
df.to_excel("file_path/model2.1.xlsx", index = False)

# df_mnact is where motor neuron activation is stored in H5 file
df = pd.read_hdf(file_path, key = 'df_mnact')
df.to_excel("file_path/mnact2.1.xlsx", index = False)

From there you may need to change the file path in the file to go directly to where you have your files saved
