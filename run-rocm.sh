name=p100

for n in `seq 5 26`
do
    for bdim in 64 128 256 512
    do
        printf "%8d%8d\n" $n $bdim
        ofile=out_${name}_${n}_${bdim}.csv
        srun rocprof --profile-from-start off --stats ./bench $n $bdim
        mv results.stats.csv > $ofile
    done
done
