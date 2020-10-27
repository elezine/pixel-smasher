20200423_082122_06_1065_3B_AnalyticMS_SR_s0118_45000.png

for file in .; do echo ${file%.png}

find /data_dir/valid_mod/ -name "*20200423_082122_06_1065_3B_AnalyticMS_SR_s0118*"

cp `find visualization/ -name "*45000.png" | shuf -n 2000` ~/data_dir/visualization &

for file in `ls`; do cp /data_dir/valid_mod/HR/x4/${file%_400000.png}.png ../valid_mod/HR/x4/; done 

find visualization/20190805_054900_71_105d_3B_AnalyticMS_SR_s0469 -name "20190805_054900_71_105d_3B_AnalyticMS_SR_s0469_[1,2,3,5,6,7,8,9,0]*.png"

# sudo chmod a+rwx $file; sudo chown ethan_kyzivat: $file; 

# for deleting file at weird checkpoint times- keeps all files with iteration that begins with 4 (corr. to final 400,000 and 400,004 iterations and "worst" 40,000 - for run 0043), and also the 5,000 iter first image and 200,000 midpoint

cd /mnt/disks/extraspace/pixel-smasher/experiments/006_ESRGAN_x4_PLANET_noPreTrain_wandb_sep28
for file in `find visualization/*/ -name "*s*_[1,3,6,7,8,9,0]*.png"`; do sudo rm $file; done & # removds most * (except 2,3,4, 5)
for file in `find visualization/*/ -name "*s*_[2][1-9][1-9]*.png"`; do sudo rm $file; done & # removes most 2*
# for file in `find visualization/*/ -name "*s*_3[0,2-9]*.png"`; do sudo rm $file; done & # removes most 3*, but keeps latest 310,000 iter # toggle this
for file in `find visualization/*/ -name "*s*_2????.png"`; do sudo rm $file; done & # removes 20,000, but keeps 200,000 midpoint
for file in `find visualization/*/ -name "*s*_5????.png"`; do sudo rm $file; done & # removes 50,000

## to rm additional in the 200,000's:
for file in `find visualization/*/ -name "*s*_2[1-9]????.png"`; do sudo rm $file; done & 

## workflow:
'''python /home/ethan_kyzivat/code/pixel-smasher/old_BasicSR/codes/scripts/extract_subimgs_single.py && bash /home/ethan_kyzivat/code/pixel-smasher/old_BasicSR/codes/utils/rand_shuf.sh && python /home/ethan_kyzivat/code/pixel-smasher/old_BasicSR/codes/scripts/generate_mod_LR_bic_parallel.py

We noticed youre using a conda environment. If you are experiencing issues with this environment in the integrated terminal, we recommend that you let the Python extension change "terminal.integrated.inheritEnv" to false in your user settings.
'''
# To update colorinterp of tif subsets snipped from ArcGis

for file in /data_dir/Scenes-shield-gt-subsets/*; do 
    gdal_translate  -colorinterp blue,green,red,alpha $file /data_dir/planet_sub/hold_mod_scenes-shield-gt-subsets/`basename $file '.tif'`.tif
done

# to create masks: use gdal_reproject_match to make pekel mask subsets, then put through normal generate_mod_LR_res.py process.

export PATH=$PATH:/home/ethan_kyzivat/code/geographic-functions
masks_dir='/data_dir/Shield_Water_Mask/Scenes-shield-gt'
out_dir='/data_dir/Shield_Water_Mask/Scenes-shield-gt-subsets'
for file in /data_dir/Scenes-shield-gt-subsets/*; do 
    IN=$masks_dir/${file:35:37}_no_buffer_mask.tif # e.g. 20170709_180516_1005_3B_AnalyticMS_SR + '_no_buffer_mask.tif'
    OUT=$out_dir/`basename $file .tif`.tif
    echo Input: $IN
    ########### uncomment for testing
    # ls $IN
    # echo OUT: $OUT
    # ls $file
    ###################

    bash gdal_reproject_match.sh $IN $OUT $file
done

# to convert from .tif to .png:
for file in /data_dir/Shield_Water_Mask/Scenes-shield-gt-subsets/*; do 
    gdal_translate -a_nodata none $file /data_dir/planet_sub/hold_mod_scenes-shield-gt-subsets_masks/`basename $file '.tif'`.png
done

# to convert to [0, 255]:
for file in /data_dir/planet_sub/hold_mod_scenes-shield-gt-subsets_masks_01/*.png; do 
    # gdal_calc.py -A $file --calc="A*255" --outfile=/data_dir/planet_sub/hold_mod_scenes-shield-gt-subsets_masks/`basename $file`
    ls $file
done

# to convert from .tif to .png (after katia work):
for file in /data_dir/planet_sub/hold_mod_scenes-shield-gt-subsets_masks_0_255_tif/*; do 
    gdal_translate -a_nodata none $file /data_dir/planet_sub/hold_mod_scenes-shield-gt-subsets_masks/`basename $file '.tif'`.png
done