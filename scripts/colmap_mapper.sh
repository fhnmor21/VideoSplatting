# run feature extraction
export LD_LIBRARY_PATH=/var/home/bazzite/Data/dotLocal/lib/


touch ./work/database.db 
colmap/build/src/colmap/exe/colmap feature_extractor \
    --database_path ./work/database.db \
    --image_path ./work/frames/ \
    --ImageReader.single_camera 1 \
    --FeatureExtraction.use_gpu 0 \
    --FeatureExtraction.num_threads 16 

# run matching
colmap/build/src/colmap/exe/colmap sequential_matcher \
    --database_path ./work/database.db \
    --SequentialMatching.overlap 20 \
    --SequentialMatching.loop_detection 1  \
    --FeatureMatching.use_gpu 0 \
    --FeatureMatching.num_threads 16

# run mapper
mkdir work/sparse
colmap/build/src/colmap/exe/colmap mapper \
  --database_path ./work/database.db \
  --image_path ./work/frames/ \
  --output_path ./work/sparse \
  --Mapper.num_threads 16 \
  --Mapper.min_num_matches 8 \
  --Mapper.init_min_num_inliers 10

# run exhaustive matcher if the above does not work.
colmap exhaustive_matcher \
    --database_path ./work/database.db \
    --FeatureMatching.use_gpu 0 \
    --FeatureMatching.num_threads 16


