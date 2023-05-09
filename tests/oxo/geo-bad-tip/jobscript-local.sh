
source /home/crduan/.bashrc

module load terachem/tip
export OMP_NUM_THREADS=1

terachem terachem_input > terachem.out &
PID_KILL=$!
molscontrol --pid $PID_KILL --config configure.json &
wait
