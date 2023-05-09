
source /home/crduan/.bashrc

module load intel/16.0.109
module load OpenMM/7.1.1
module load mpich2/1.4.1p1
module load cuda/10.0
export PATH=/home/crduan/src/tc-production-fang-stable/build/bin:$PATH
export LD_LIBRARY_PATH=/home/crduan/src/tc-production-fang-stable/build/lib:$LD_LIBRARY_PATH
export TeraChem=/home/crduan/src/tc-production-fang-stable/build
export NBOEXE=/home/crduan/src/tc-production-fang-stable/nbo6/bin/nbo6.i4.exe

export OMP_NUM_THREADS=1

terachem terachem_input > terachem.out &
PID_KILL=$!
molscontrol --pid $PID_KILL --config configure.json &
wait
