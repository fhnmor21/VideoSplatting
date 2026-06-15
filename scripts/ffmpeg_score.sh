ffmpeg -i ../video/A001_06091006_C002.mp4 -vf select='not(not(scene))',metadata=print:key=lavfi.scene_score -f null -
