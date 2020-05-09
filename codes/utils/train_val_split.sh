# bash script to move files from generate_mod_LR_bic output to train / val folders

VALSIZE=1000 # number of validation files (rest for training)
#shuf all1.txt -n 1000 > valid_shuf1.txt
basepath=/data_dir/planet_sub_LR_cal
suffix=x8

for folder in Bic  HR  LR; do
    input=$basepath/$folder/$suffix
    echo $input
    for mv_folder in /data_dir/valid_mod_cal/$folder/$suffix; do # /data_dir/train_mod_cal
        echo mv folder: $mv_folder
        #mkdir -p $mv_folder
        #cat valid_shuf.txt | while read file; do mv $input/$file $mv_folder/$file; done
        # # mv $input/*.png /data_dir/train_mod_cal/$folder/$suffix/$file # too long...
        mkdir -p train_mod_cal/$folder/$suffix
        ls $input  | while read file; do mv $input/$file /data_dir/train_mod_cal/$folder/$suffix/$file; done
    done
done
