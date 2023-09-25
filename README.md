
# Discription

This repository contains developed programmable interfaces for autonomously steering an electrochemistry workstation from a remote computing system over an ecosystem infrastructure. The developed interfaces are Python-based to control Bio-Logic potentiostat and run Cyclic Voltammetry (CV) experiments on a connected electrochemical cell. The interfaces also control a custom electrochemical setup to fill the cell with a liquid and push gas.

We utilized the developed interfaces to develop an [electrochemistry workflow](https://github.com/aneesalnajjar/electrochemistry/blob/main/README.md#electrochemistry-workflow) to test the normality of CV measurements using machine learning.

More details about the cross-facility electrochemistry workflow and ML for electrochemistry workflow are available at:

  **1. A. Al-Najjar, N. S. V. Rao,  C. Bridges and S. Dai, "Cross-Facility Orchestration of Electrochemistry Experiments and Computations", In 2023 5th Annual Workshop on Extreme-scale Experiment-in-the-Loop Computing (XLOOP), Denver, CO, USA, 2023.**

  **2. A. Al-Najjar, N. S. V. Rao, C. Bridges, S. Deng, Normality of I-V measurements using ML, IEEE International Conference eScience, 2-pager and poster, October 9-13, Limassol, Cyprus.**

## Developer:
Anees Al-Najjar,

Oak Ridge National Laboratory (ORNL),

Oak Ridge, TN, 37830, USA

# Prerequitise

1. Anaconda installed on the remote computing systems and control agent.
2. EC-Lab Development Package for Bio-Logic SP200 potentiostat firmware installed on C drive of the control agent. Product info https://www.biologic.net/products/sp-200/
3. J-Kem single-board (back-end) hardware and firmware connected to the J-Kem setup. J-Kem setup of instruments includes MFC, Fraction collector, Syringe and peristaltic pumps, temperature controller and monitor, polyScience chiller, pH probe, and Electrode Module.
Products info https://www.jkem.com/, and email1, and email2
4. Interconnect the electrochemistry workstation instruments into an ecosystem with remote computing system

Note:
1. The workflow modules can work with 2 and 3, one of them, or a partial set of 3.
2. So far, only the CV technique is available as part of developed autonomous and remote steering modules of electrochemical testing with potentiostat.

#################################################################################
#################################################################################

# Electrochemistry Cross facility testing Ecosystem.
 
The Ecosystem consists of an electrochemistry workstation connected to a control agent computer at science facility, which is interconnected to a remote (high-performance) computing system available at a different facility.
The electrochemistry workstation includes Bio-Logic SP200 potentiostat to control an electrochemical cell, and a J-Kem custom setup of MFC, Fraction collector, Syringe and peristaltic pumps, temperature controller and monitor, polyScience chiller, pH probe, and Electrode Module. The setup is connected via serial ports to the J-Kem single-board computer that runs (back-end) vendor control firmware.
The electrochemical cell is fed with a solution and gas via the J-Kem setup to run Cyclic Voltammetry (CV) test. 
The potentiostat and J-Kem single-board computer are controlled via developed Python APIs embedded in the control agent.
The Python-based APIs at the control agent are wrapped as Pyro server objects to be remotely called across the ecosystem network from the remote computing system to enable cross-facility autonomous instruments steering, and measurement transfer and analysis.
	
# Electrochemistry Workflow

Initially, as a manual step, the workflow requires initializing the experimental setup of the instruments, including filling the fraction collector vials with a solution. 
Then, the workflow modules are run in server/client mode. The server modules are run at the control agent and the client modules are run from the remote computing.
The server modules are triggered at the control agent to be run in a daemon to expose the Pyro server objects for communication across the network. They are called by Pyro client applications executed and orchestrated as part of autonomous workflow from a Jupyter notebook at the remote computing system.

The workflow includes:
  * Activating the J-Kem setup to fill the electrochemical cell.
  * Activating potentiostat to run the CV experiment.
  * Collecting measurements at the control agent.
  * Making them available at the remote system for analysis. Different methods are used for streaming measurements, including via control channel over Pyro, or via a dedicated data channel, such as MS One Drive or a file-sharing technique.

So far, the analysis modules available in this repo are to check the CV profile normality using machine learning, particularly to examine the profiles if they are "normal" or "invalid" due to the disconnection of an electrode or running out of solution at the cell.
#################################################################################
#################################################################################

# To create a virtual environment on (client(c)/server(s)) system
conda create -y --name acl_venv_<c|s> python=3.9
conda install --force-reinstall -y -q --name acl_venv_<c|s> -c conda-forge --file requirements_<win/lnx>.txt

conda activate acl_venv_<c|s>
#.....
conda deactivate

Note:
1. requirements_win.txt is for the client and server if both are Windows-based systems.
2. requirements_lnx.txt if the client is a Linux-based system.

# To access jupyter-notebook on the client (Linux (e.g., dgx system))
### on Linux (dgx system)
cd /mnt/acl_ecosystem/Workflow_dependencies/
conda activate acl_env
jupyter notebook --no-browser --port=8080

### on client system terminal
ssh -L 8080:localhost:8080 <user>@phoenix

### on client system web browser
http://localhost:8080/tree?token=< token >

#################################################################################
#################################################################################
