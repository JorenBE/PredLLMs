for run in 1
do
for representation in SMILES
do
for epoch in 4
do
for train_size in 400
do
for target in y_bin
do
echo cmc run $run Epoch $epoch Train size $train_size $representation $target
python run_experiment.py $train_size $run EleutherAI/gpt-j-6b $epoch $representation $target train_polymers.csv 50

done
done
done
done
done