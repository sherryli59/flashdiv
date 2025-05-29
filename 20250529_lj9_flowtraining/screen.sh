for t in 1.0 2.0 5.0
do
for lr in 0.0001 0.00001
do
sbatch submit.sh $t $lr
done
done
