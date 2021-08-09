rm -rf logs
rm -rf sim_logs
mkdir logs
mkdir sim_logs
./build.sh

OMP_NUM_THREADS=1 nohup python3 train.py &> flocking1.out &