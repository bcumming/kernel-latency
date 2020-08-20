for f in *.csv
do
    sed -i '/us,us/d' $f
done
