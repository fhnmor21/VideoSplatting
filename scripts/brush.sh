./brush/target/release/brush \
  /path/to/colmap_dataset \
  --with-viewer \
  --total-train-iters 10000000 \
  --refine-every 1000 \
  --max-resolution 4096 \
  --sh-degree 3 \
  --export-every 5000 \
  --export-path "./{dataset}_exports/" \
  --export-name "export_{iter}.ply"
