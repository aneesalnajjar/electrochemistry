import numpy as np
import random
import os
from biologic_sp200_config import SP200_Technique_params

#########################################################
########################################################
# plots saving/showing configurations
save_file_flag=True
show_IV_plot=True

########################################################
########################################################
# Server connection ip and port
ipAddressServer = '0.0.0.0'
connectionPort='0000'

########################################################
########################################################
# Data transfer methods and related paths
profiles_via_data_channel=False # Default is false

# Note: Paths may be different based on the channel sent through.
# For example, the testing data paths on the client and server are different if the data_channel is used.
# The measurements collected on the server/instrument controller should point to its local directory. 
# The client runs on a remote system and accesses the measurements across a mounted directory whose path is different. 

####################
# Paths at the client side if measurements are sent via the control Channel (using Pyro)
EoT_Classifier_Path = "./Workflow_dependencies/ml_models_dir"
training_data_path = "./Workflow_dependencies/training_profiles"
testing_data_path="./Workflow_dependencies/testing_profiles"
# Create the directories if they do not exist. 
# Make sure you place the training samples in the training_profiles for the first time.
os.makedirs(EoT_Classifier_Path, exist_ok=True) # create EoT_Classifier_Path if it is not exist
os.makedirs(training_data_path, exist_ok=True) # create training_data_path if it is not exist
os.makedirs(testing_data_path, exist_ok=True) # create testing_data_path if it is not exist


######################
# Paths if measurements are sent via the data Channel over One Drive

# usr='xxx'

# EoT_Classifier_Path = f"C:\\Users\\{usr}\\OneDrive\\acl_sp_200\\Workflow_dependencies\\ml_models_dir"
# training_data_path=f"C:\\Users\\{usr}\\OneDrive\\acl_sp_200\\Workflow_dependencies\\training_profiles"
# testing_data_path=f"C:\\Users\\{usr}\\OneDrive\\acl_sp_200\\Workflow_dependencies\\testing_profiles"

# # Create the directories if they do not exist. 
# # Make sure you place the training samples in the training_profiles for the first time.
# os.makedirs(EoT_Classifier_Path, exist_ok=True) # create EoT_Classifier_Path if it is not exist
# os.makedirs(training_data_path, exist_ok=True) # create training_data_path if it is not exist
# os.makedirs(testing_data_path, exist_ok=True) # create testing_data_path if it is not exist

#####################

# Paths if the measurements are sent via the data Channel using file sharing technique (CIFS)
# These paths are on the client side, which is a Linux DGX system 
# Note this Jupyter notebook runs on DGX.

# EoT_Classifier_Path = "/mnt/acl_ecosystem/Workflow_dependencies/ml_models_dir"
# training_data_path="/mnt/acl_ecosystem/Workflow_dependencies/training_profiles"
# testing_data_path= "/mnt/acl_ecosystem/Workflow_dependencies/testing_profiles"
# #The result path on the server side (windows control agent connected to the potentiostat and J-KEM setup)
# testing_data_path_server="C:/acl_ecosystem/Workflow_dependencies/testing_profiles"


###############################################################################################
###############################################################################################

# ML Configuration
v_probe=np.linspace(SP200_Technique_params['technique']['Voltage_step_E']['E2'],\
                    SP200_Technique_params['technique']['Voltage_step_E']['E1'],\
                    10).reshape(-1, 1) # 1-1 )sholud be aligned with scan rate range

# Classifier file name
EoT_Classifier = "clf.pckl"
