# Training on UBC ECE Machines via SSH

**VPN access instructions:** https://help.ece.ubc.ca/How_To_Use_VPN

## Summary

For some Phase 3 experiments, we used UBC ECE lab machines instead of training only on a local computer. This provided access to stronger shared compute resources and allowed us to run longer RF-DETR training jobs remotely through SSH and Jupyter Lab.

This setup was especially useful for experiments that were too slow or inconvenient to run entirely on a personal machine, while still keeping the workflow flexible and close to a normal notebook-based development process.

This README explains:

- why we used the UBC machines
- how we connected to them
- how we set up the Python environment
- how we launched Jupyter Lab remotely
- how we kept long jobs running with `tmux`
- how we moved files between the local machine and the remote machine
- what notebooks we ran there
- what we learned from those experiments

---

## Why we used UBC's machines

We used UBC's machines because some of the Phase 3 RF-DETR training runs were long enough that running them only on a local workstation was inconvenient.

The UBC setup gave us a practical middle ground:

- access to stronger remote hardware
- ability to run notebook-based experiments over SSH
- ability to leave long jobs running remotely
- easier experimentation without depending entirely on one local machine staying on for the full duration

This was useful for testing alternate training strategies before deciding which ones should remain part of the final pipeline.

---

## Experiments run on the UBC machine

We used the UBC machine to run additional RF-DETR training notebooks, including:

- `rfdetr_training_rgb_video.ipynb`
- `rfdetr_training_infrared_isolated.ipynb`

These experiments were exploratory and were used to evaluate whether alternative data sources or training strategies improved performance.

### 1. RGB video training

This stage extended the RGB image baseline by training RF-DETR Nano on **VisDrone video data** rather than only VisDrone still images.

The idea was simple: video provides many more labeled RGB frames, so it looked like a natural way to expand the training set.

However, the VisDrone-VID annotation format is not identical to the still-image dataset. In particular:

- the video labels include an explicit `ignored_regions` class
- only rows with `score = 1` are treated as valid training annotations
- ignored regions, invalid rows, and excluded source classes are skipped during conversion

So this dataset is not just “more RGB data.” It uses stricter supervision rules and a somewhat different annotation structure.

Although the video experiment added many more frames, adjacent frames are often highly similar. In practice, that means the model sees a large amount of temporally redundant content. This appears to have made the training set less diverse than the raw frame count suggests.

The observed validation behavior supported that interpretation:

- performance peaked early
- validation metrics later trended downward
- the training behavior looked more like overfitting than sustained improvement

Because of that, the RGB video checkpoint was **not** adopted into the final pipeline. It is treated as a useful negative result showing that adding many highly correlated video frames does not necessarily improve generalization.

### 2. Isolated infrared training

We also ran the **isolated infrared training** experiment on the UBC machine.

This notebook trained RF-DETR Nano directly on DroneVehicle infrared data without continuing from the enhanced RGB checkpoint. The purpose was to measure how strong a clean infrared-only detector could be when trained as its own modality-specific model.

This experiment was important because it allowed us to compare:

- infrared-only training from scratch
- infrared continuation training from the enhanced RGB checkpoint

That comparison helped clarify whether RGB-to-infrared transfer was actually beneficial in this project setup.

---

## Before connecting: VPN

If you are off campus, connect to the UBC ECE VPN first:

https://help.ece.ubc.ca/How_To_Use_VPN

This may be required before SSH access works, depending on your network and the machine you are connecting to.

---

## Connecting to the UBC machine

From your **local machine**, connect with SSH:

```bash
ssh your_username@your_ubc_machine
```
If you want to use Jupyter Lab remotely through your browser, use local port forwarding:
```
ssh -L 8888:localhost:8888 your_username@your_ubc_machine
```
Then open this in your local browser:
```
http://localhost:8888/lab
```

## Python environment setup
We used a dedicated Python 3.11.9 environment through Miniconda.

### 1. Install Miniconda
On the remote UBC machine:
```
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```
When prompted for the install path, choose something like:
```
/home/your_username/miniconda3
```

### 2. Activate Miniconda
```
source ~/miniconda3/bin/activate
```

### 3. Create the Python 3.11.9 environment

```conda create -n py311 python=3.11.9 -y
conda activate py311
python --version
```
### 4. Install project dependencies

From the directory containing your notebook requirements:
```
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

## Register the environment as a Jupyter kernel

To make the environment available inside Jupyter Lab:
```
python -m ipykernel install --user --name=rfdetr-env --display-name "Phase 3 RF-DETR Env"
```

This allows the training notebooks to run with the correct environment from the Jupyter interface.

## Starting Jupyter Lab on the remote machine

After activating the environment:
```
jupyter lab --no-browser --port=8888
```

Then, from your local machine, forward the port:
```
ssh -L 8888:localhost:8888 your_username@your_ubc_machine
```

Finally, open:
```
http://localhost:8888/lab
```

This lets you use the remote UBC machine through a local browser while the actual notebook execution happens on the remote system.

## Keeping long training jobs alive with tmux

For long-running jobs, we used tmux so the process would continue even if the local terminal closed or the laptop turned off.

### Start a tmux session

SSH into the remote machine:
```
ssh your_username@your_ubc_machine
```

Then run:
```
tmux
source ~/miniconda3/bin/activate
conda activate py311
jupyter lab --no-browser --port=8888
```

### Detach from tmux

To leave the session running:

- press `Ctrl+b`
- then press `d`

At that point, the remote session continues running and you can close your local terminal.

### Reattach later

To reconnect Jupyter in the browser again:
```
ssh -L 8888:localhost:8888 your_username@your_ubc_machine
```

Then open:
```
http://localhost:8888/lab
```

To reattach to the terminal session itself:
```
ssh your_username@your_ubc_machine
tmux attach
```

That lets you return to the exact terminal where Jupyter Lab or another long-running command was launched.

## Checking whether a notebook is still running

To see whether a notebook execution process is still active on the remote machine:
```
ps -ef | grep _runner.py
```

This is useful when you want to confirm whether a training notebook is still executing in the background.

## Copying files between local and remote machines
### Copy a local file to the remote machine
```
scp FILE your_username@your_ubc_machine:~
```

### Move a file on the remote machine
```
mv file.txt PATH
```

### Check the size of a folder
```
du -sb FOLDER
```

These commands were useful for moving notebooks, logs, metadata, and intermediate files during experimentation.

## Recommended workflow

A typical workflow looked like this:

1. connect to the VPN if needed
2. SSH into the remote UBC machine
3. activate the conda environment
4. start Jupyter Lab inside a tmux session
5. forward port 8888 from the local machine
6. open Jupyter Lab in the browser
7. run the desired training notebook remotely
8. detach from tmux if the run needs to continue unattended
9. reconnect later to inspect logs or results

This kept the workflow very similar to local notebook development while shifting the actual computation onto the remote machine.

## What we found

The UBC machine experiments were useful because they helped us evaluate training directions before committing them to the final Phase 3 pipeline.

### RGB video training result

The RGB video training experiment **did** not outperform the stronger RGB image-based training path.

Although it introduced many more frames, those frames were highly correlated across time. That reduced the effective diversity of the training signal and appears to have encouraged overfitting. Validation behavior peaked early and then declined, which made this approach less stable and less attractive than the enhanced RGB image training pipeline.

Because of that, the RGB video model checkpoint was **not carried forward** into the final training workflow.

### Isolated infrared training result

The isolated infrared experiment was more valuable for the final analysis. It provided a clean infrared-only baseline and helped compare direct infrared learning against infrared-over-RGB continuation training.

That comparison was important for understanding whether RGB pretraining actually helped in the target modality, rather than simply assuming it would.

### Conclusion

Using UBC's machines over SSH gave us a practical remote training workflow for Phase 3 experiments. It allowed us to:

- run longer RF-DETR notebook experiments remotely
- keep jobs alive using tmux
- access notebooks through Jupyter Lab in a browser
- test alternate training strategies without depending fully on a local machine

The experiments run there helped refine the final pipeline:

- RGB video training was tested but not adopted because of worse validation behavior and weaker generalization
- isolated infrared training was kept as an important comparison point for the final infrared training story

This made the UBC SSH workflow a useful part of the experimental development process, even though not every model trained there was promoted into the final system.