gdown "https://drive.google.com/uc?id=1nNuEJdNuGV3iHwvPwS7FfZQCJ6ZZO3PD"    

mkdir -p eval_datas/dstc10_format
unzip dstc10.zip 
mv human_evaluation_data/* eval_datas/dstc10_format/
rmdir human_evaluation_data
rm dstc10.zip
