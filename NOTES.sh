20200423_082122_06_1065_3B_AnalyticMS_SR_s0118_45000.png

for file in .; do echo ${file%.png}

find /data_dir/valid_mod/ -name "*20200423_082122_06_1065_3B_AnalyticMS_SR_s0118*"

cp `find visualization/ -name "*45000.png" | shuf -n 2000` ~/data_dir/visualization &

for file in `ls`; do cp /data_dir/valid_mod/HR/x4/${file%_400000.png}.png ../valid_mod/HR/x4/; done 

find visualization/20190805_054900_71_105d_3B_AnalyticMS_SR_s0469 -name "20190805_054900_71_105d_3B_AnalyticMS_SR_s0469_[1,2,3,5,6,7,8,9,0]*.png"

# sudo chmod a+rwx $file; sudo chown ethan_kyzivat: $file; 

# for deleting file at weird checkpoint times- keeps all files with iteration that begins with 4 (corr. to final 400,000 and 400,004 iterations and "worst" 40,000 - for run 0043), and also the 5,000 iter first image and 200,000 midpoint

for file in `find visualization/*/ -name "*s*_[1,6,7,8,9,0]*.png"`; do sudo rm $file; done & # removds most * (except 2,3,4, 5)
for file in `find visualization/*/ -name "*s*_[2][1-9][1-9]*.png"`; do sudo rm $file; done & # removes most 2*
for file in `find visualization/*/ -name "*s*_3[0,2-9]*.png"`; do sudo rm $file; done & # removes most 3*, but keeps latest 310,000 iter # toggle this
for file in `find visualization/*/ -name "*s*_2????.png"`; do sudo rm $file; done & # removes 20,000, but keeps 200,000 midpoint
for file in `find visualization/*/ -name "*s*_5????.png"`; do sudo rm $file; done & # removes 50,000