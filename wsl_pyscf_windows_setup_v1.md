# WSL + PySCF Environment Setup for Windows

This guide sets up a clean Linux-based PySCF environment on Windows using **WSL 2**, **Ubuntu 24.04 LTS**, and **Miniforge/conda**. This is generally more reliable than trying to use PySCF directly in native Windows Python, because PySCF provides precompiled Linux wheels that work on WSL.

Last checked: 2026-05-20

## 0. Overview

Recommended stack:

```text
Windows 10/11
└── WSL 2
    └── Ubuntu 24.04 LTS
        └── Miniforge / conda
            └── Python 3.12 environment
                └── PySCF
```

The commands below assume a normal x86_64 Windows laptop/desktop. If you are using a Windows-on-ARM machine, replace the Miniforge installer with the Linux `aarch64` installer.

---

## 1. Install WSL 2 and Ubuntu

Open **PowerShell as Administrator** and run:

```powershell
wsl --install -d Ubuntu-24.04
```

Restart Windows if prompted.

If the command does not recognize `Ubuntu-24.04`, list available distributions first:

```powershell
wsl --list --online
```

Then install an available Ubuntu version, for example:

```powershell
wsl --install -d Ubuntu
```

After installation, check that Ubuntu is using WSL 2:

```powershell
wsl -l -v
```

Expected style of output:

```text
  NAME            STATE           VERSION
* Ubuntu-24.04    Running         2
```

If the version is `1`, convert it to WSL 2:

```powershell
wsl --set-version Ubuntu-24.04 2
```

If your distribution name is just `Ubuntu`, use:

```powershell
wsl --set-version Ubuntu 2
```

You can also update WSL itself:

```powershell
wsl --update
```

---

## 2. First Ubuntu launch

Open **Ubuntu** from the Windows Start menu or run:

```powershell
wsl ~ -d Ubuntu-24.04
```

If your distro name is just `Ubuntu`, run:

```powershell
wsl ~ -d Ubuntu
```

On first launch, Ubuntu will ask you to create a Linux username and password. This password is used for `sudo` commands inside Ubuntu. It does not need to match your Windows password.

---

## 3. Update Ubuntu packages

Inside the Ubuntu terminal, run:

```bash
sudo apt update
sudo apt upgrade -y
sudo apt install -y build-essential git curl wget ca-certificates
```

Optional, but useful if a Python package ever falls back to a source build:

```bash
sudo apt install -y gfortran pkg-config libopenblas-dev liblapack-dev
```

---

## 4. Use the WSL filesystem for projects

For better performance, keep Python projects inside the Linux filesystem, for example under `~/workspace`, not under `/mnt/c/...`.

Create a workspace:

```bash
mkdir -p ~/workspace
cd ~/workspace
```

Recommended:

```text
/home/<linux-user>/workspace/my_project
```

Avoid for heavy Python/conda workflows:

```text
/mnt/c/Users/<windows-user>/Desktop/my_project
```

You can still access Linux files from Windows Explorer using:

```text
\\wsl.localhost\Ubuntu-24.04\home\<linux-user>
```

or, for older WSL versions:

```text
\\wsl$\Ubuntu-24.04\home\<linux-user>
```

---

## 5. Install Miniforge inside WSL

Run the following inside Ubuntu, not PowerShell:

```bash
cd ~
curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
```

During installation:

1. Press `Enter` to read the license.
2. Type `yes` to accept.
3. Accept the default install path, usually:

```text
/home/<linux-user>/miniforge3
```

4. When asked whether to initialize conda, type:

```text
yes
```

Close and reopen Ubuntu, or run:

```bash
source ~/.bashrc
```

Check conda/mamba:

```bash
conda --version
mamba --version
```

Set strict conda-forge channel priority:

```bash
conda config --set channel_priority strict
```

---

## 6. Create a PySCF environment

Create a dedicated environment. Python 3.12 is a conservative choice for scientific Python packages.

```bash
mamba create -n pyscf-env python=3.12 -y
mamba activate pyscf-env
```

Upgrade pip:

```bash
python -m pip install --upgrade pip setuptools wheel
```

Install PySCF using the PySCF-recommended binary-wheel route:

```bash
python -m pip install --prefer-binary pyscf
```

Install common scientific/Jupyter tools:

```bash
python -m pip install numpy scipy matplotlib pandas jupyterlab ipykernel
```

Register the environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name pyscf-env --display-name "Python (pyscf-env)"
```

---

## 7. Verify PySCF

Run:

```bash
python - <<'PY'
import pyscf
from pyscf import gto, scf

print("PySCF version:", pyscf.__version__)

mol = gto.M(
    atom="H 0 0 0; H 0 0 0.74",
    basis="sto-3g",
    verbose=0,
)

mf = scf.RHF(mol).run()
print("H2 RHF energy:", mf.e_tot)
PY
```

Expected behavior:

```text
PySCF version: ...
H2 RHF energy: around -1.1167 Hartree
```

A slightly different final digit is normal.

---

## 8. Start JupyterLab

Inside Ubuntu:

```bash
mamba activate pyscf-env
cd ~/workspace
jupyter lab --no-browser
```

Jupyter will print a URL similar to:

```text
http://localhost:8888/lab?token=...
```

Open that URL in your Windows browser.

---

## 9. Use VS Code with WSL

Install VS Code on Windows, then install the **WSL** extension in VS Code.

From Ubuntu:

```bash
cd ~/workspace
code .
```

This opens VS Code connected to the WSL filesystem. In notebooks or Python files, select the kernel/interpreter named:

```text
Python (pyscf-env)
```

---

## 10. Daily usage

Open Ubuntu and activate the environment:

```bash
mamba activate pyscf-env
cd ~/workspace
```

Run Python:

```bash
python
```

Run JupyterLab:

```bash
jupyter lab --no-browser
```

Update PySCF later:

```bash
mamba activate pyscf-env
python -m pip install --upgrade pyscf
```

---

## 11. Alternative: install PySCF from conda-forge

The pip route above is the PySCF-recommended default for non-developers. If you prefer a fully conda-managed environment, you can instead run:

```bash
mamba create -n pyscf-conda python=3.12 pyscf numpy scipy matplotlib pandas jupyterlab ipykernel -c conda-forge -y
mamba activate pyscf-conda
python -m ipykernel install --user --name pyscf-conda --display-name "Python (pyscf-conda)"
```

Do not mix the two approaches unnecessarily. For a single environment, prefer either:

```text
pip-installed PySCF inside conda environment
```

or:

```text
conda-forge-installed PySCF
```

---

## 12. Troubleshooting

### `wsl --install` fails or hangs

Try updating WSL:

```powershell
wsl --update
```

Then list installable distributions:

```powershell
wsl --list --online
```

Install one explicitly:

```powershell
wsl --install -d Ubuntu-24.04
```

If Ubuntu 24.04 is unavailable on your system, use:

```powershell
wsl --install -d Ubuntu
```

### WSL says virtualization is disabled

Enable virtualization in your BIOS/UEFI settings. The option is usually named one of:

```text
Intel VT-x
Intel Virtualization Technology
AMD-V
SVM Mode
Virtualization Technology
```

Then reboot Windows and try again.

### PySCF installation tries to build from source

First check that you are using the intended environment:

```bash
which python
python --version
python -m pip --version
```

Then retry with binary preference:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install --prefer-binary pyscf
```

If it still tries to compile, install the optional build dependencies:

```bash
sudo apt install -y build-essential gfortran pkg-config libopenblas-dev liblapack-dev
```

Then retry:

```bash
python -m pip install --prefer-binary pyscf
```

### Jupyter cannot find the PySCF environment

Activate the environment and register the kernel again:

```bash
mamba activate pyscf-env
python -m ipykernel install --user --name pyscf-env --display-name "Python (pyscf-env)"
```

Restart JupyterLab and select:

```text
Python (pyscf-env)
```

### Conda is not found after Miniforge installation

Run:

```bash
source ~/.bashrc
```

If that does not work, initialize conda manually:

```bash
~/miniforge3/bin/conda init bash
source ~/.bashrc
```

### Performance is slow

Check your current directory:

```bash
pwd
```

If it starts with `/mnt/c/`, move your project into the WSL filesystem:

```bash
mkdir -p ~/workspace
cp -r /mnt/c/Users/<windows-user>/path/to/project ~/workspace/
cd ~/workspace/project
```

---

## 13. Minimal reproducibility record

For debugging or sharing with collaborators, record:

```bash
wsl.exe -l -v
```

from PowerShell, and inside Ubuntu:

```bash
uname -a
lsb_release -a
which python
python --version
python -m pip show pyscf
conda env export --from-history
```

Save the output in a file:

```bash
conda env export --from-history > pyscf-env-history.yml
```

To recreate the environment later:

```bash
mamba env create -f pyscf-env-history.yml
```

---

## References

- Microsoft WSL installation documentation: https://learn.microsoft.com/en-us/windows/wsl/install
- Ubuntu WSL 2 installation documentation: https://documentation.ubuntu.com/wsl/latest/howto/install-ubuntu-wsl2/
- Microsoft WSL filesystem performance guidance: https://learn.microsoft.com/en-us/windows/wsl/filesystems
- PySCF installation documentation: https://pyscf.org/user/install.html
- conda-forge Miniforge installer documentation: https://conda-forge.org/download/
- conda-forge PySCF package page: https://anaconda.org/conda-forge/pyscf
