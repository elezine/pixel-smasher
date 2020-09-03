# bash script to move files from generate_mod_LR_bic output to train / val folders

# use after: find . -type f -iname "*.png"  |  cut -c 3- > ../all10.txt

# VALSIZE=1000 # number of validation files (rest for training)
#shuf all1.txt -n 1000 > valid_shuf1.txt
basepath=/data_dir/train_mod/
basepath_out=/data_dir/valid_mod # hold valid
shuf_list=/data_dir/planet_sub/valid_shuf10.txt # hold valid
# all_list=/data_dir/planet_sub/all10.txt # hold valid
suffix=x4

for folder in Bic  HR  LR; do
    input=$basepath/$folder/$suffix
    echo $input
    for mv_folder in $basepath_out/$folder/$suffix; do # /data_dir/train_mod_cal
        echo mv folder: $mv_folder
        # mkdir -p $mv_folder
        cat $shuf_list | while read file; do mv $input/$file $mv_folder/$file; done
        # # mv $input/*.png /data_dir/train_mod_cal/$folder/$suffix/$file # too long...
        # mkdir -p train_mod_cal/$folder/$suffix
        # ls $input  | while read file; do mv $input/$file /data_dir/train_mod_cal/$folder/$suffix/$file; done
    done
done
