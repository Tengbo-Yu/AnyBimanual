# INSTALLATION

To install the dependencies execute the `scripts/install_dependencies.sh`

```bash
scripts/install_conda.sh # Skip this step if you already have conda installed.
scripts/install_dependencies.sh
```

Please see the [README](README.md) for a quick start instruction.


Alternatively, you can follow the detailed instructions to setup the software from scratch

#### 1. Environment

Install miniconda if not already present on the current system.You can use `scripts/install_conda.sh` for this step:
```bash
sudo apt install curl 

curl -L -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh 
./Miniconda3-latest-Linux-x86_64.sh

SHELL_NAME=`basename $SHELL`
eval "$($HOME/miniconda3/bin/conda shell.${SHELL_NAME} hook)"
conda init ${SHELL_NAME}
conda install mamba -c conda-forge
conda config --set auto_activate_base false
```

Next, create the rlbench environment and install the dependencies

```bash
conda create -n rlbench python=3.8
conda activate rlbench
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

#### 2. PyRep and Coppelia Simulator

Follow instructions from the [PyRep fork](https://github.com/markusgrotz/PyRep); reproduced here for convenience:

PyRep requires version **4.1** of CoppeliaSim. Download: 
- [Ubuntu 20.04](https://www.coppeliarobotics.com/files/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz)

Once you have downloaded CoppeliaSim, you can pull PyRep from git:

```bash
cd third_party
cd PyRep
```

Add the following to your *~/.bashrc* file: (__NOTE__: the 'EDIT ME' in the first line)

```bash
export COPPELIASIM_ROOT=<EDIT ME>/PATH/TO/COPPELIASIM/INSTALL/DIR
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
```

Remember to source your bashrc (`source ~/.bashrc`) or 
zshrc (`source ~/.zshrc`) after this.

**Warning**: CoppeliaSim might cause conflicts with ROS workspaces. 

Finally install the python library:

```bash
pip install -e .
```

You should be good to go!
You could try running one of the examples in the *examples/* folder.

#### 3. RLBench

PerAct^2 uses the [RLBench fork](https://github.com/markusgrotz/RLBench). 

```bash
cd third_party
cd RLBench
pip install -e .
```

For [running in headless mode](https://github.com/MohitShridhar/RLBench/tree/peract#running-headless), tasks setups, and other issues, please refer to the [official repo](https://github.com/stepjam/RLBench).

#### 4. YARR

PerAct^2 uses the [YARR fork](https://github.com/markusgrotz/YARR).

```bash
cd third_party
cd YARR
pip install -e .
```

#### 5. pytorch3d

```bash
cd third_party
cd pytorch3d
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
pip install -e .
```

#### 6. wandb

```bash
pip install wandb==0.14.0
```






