# bash script to move files from generate_mod_LR_bic output to train / val folders
# inputs
BASE_DIR=/data_dir/planet_sub

# choose random subsets of files to move
NUM_IMAGES=$(find $BASE_DIR -maxdepth 1 -name "*.png" | wc -l) # modified to use find instead of ls, which fails for large lists
echo "Number of images total:" $NUM_IMAGES
VALSIZE=$(( $NUM_IMAGES * 150 / 1000 )) # number of validation files + holdout files
HOLDSIZE=$(( $NUM_IMAGES * 1 / 1000 )) # number of holdout files (rest for training)
echo "Number of images for validation:" $VALSIZE
echo "Number of images for holdout:" $HOLDSIZE

# compute filenamesmes
find $BASE_DIR -name "*.png" | cut -d'/' -f4 > all10.txt # TODO make this a less sloppy fix by dynamically referencing file names
shuf all10.txt -n $VALSIZE > valid_shuf10.txt
shuf valid_shuf10.txt -n $HOLDSIZE > hold_shuf10.txt

# mkdirs
mkdir -p $BASE_DIR/train_mod
mkdir -p $BASE_DIR/valid_mod
mkdir -p $BASE_DIR/hold_mod

## move files
cat hold_shuf10.txt | while read file; do
    mv $BASE_DIR/$file $BASE_DIR/hold_mod
done
cat valid_shuf10.txt | while read file; do
    mv $BASE_DIR/$file $BASE_DIR/valid_mod # expect Errors because some files have already been moved
done
for file in $BASE_DIR/*.png; do
    mv $file $BASE_DIR/train_mod
done
echo Moved files.


# basepath=/data_dir/train_mod


# for folder in Bic  HR  LR; do
#     input=$basepath/$folder/$suffix
#     echo $input
#     for mv_folder in /data_dir/valid_mod_cal/$folder/$suffix; do # /data_dir/train_mod_cal
#         echo mv folder: $mv_folder
#         #mkdir -p $mv_folder
#         #cat valid_shuf.txt | while read file; do mv $input/$file $mv_folder/$file; done
#         # # mv $input/*.png /data_dir/train_mod_cal/$folder/$suffix/$file # too long...
#         # mkdir -p train_mod_cal/$folder/$suffix
#         # ls $input  | while read file; do mv $input/$file /data_dir/train_mod_cal/$folder/$suffix/$file; done
#     done
# done
