name=p100

for n in `seq 5 22`
do
    for bdim in 64 128 256 512
    do
        printf "%8d%8d\n" $n $bdim
        ofile=out_${name}_${n}_${bdim}.csv
        srun nvprof --profile-from-start off --normalized-time-unit us --csv ./bench $n $bdim >& tmp
        awk '/Profiling result/{p=1;next}p' tmp > $ofile
    done
done
