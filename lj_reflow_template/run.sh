nparticles=16
logname="lj${nparticles}2d_mala_periodic_1.0"
python generate_raw_data.py \
    --logname $logname \
    --run 0 \
    --save_every 100 \
    --nparticles $nparticles \
    --dim 2 \
    --kT 1 \
    --sampler MALA \
    --step_size 0.001 \
    --num_steps 20000 \
    --burn_in 10000 \
    --batch_size 1000 \
    --spring_constant 0.001 \
    --adaptive_step_size \
    --periodic

