# SINDy Autoencoder for a Simple Pendulum

## Pendulum simulations
Run the simulations:

`> cd pendulum/simulations/`\
`> bash run_sims.sh`

which create the output files at `pendulum/simulations/sims`.

Then, create the binary files with the image data, by executing

`> python gen_pendulum_img.py`

which generates the binaries `X.npy`, `Xdot.npy`, `Xddot.npy`, and `Y.npy`. The first three are the image data for the pendulum and its time derivatives, while `Y.npy` stores the actual value for the angle.  

## SINDy autoencoder
The code for creating and training the autoencoder using the data stored in the `.npy` files is located in the notebook `pendulum/sindy_autoencoder.ipynb`
