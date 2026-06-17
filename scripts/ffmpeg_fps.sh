ffmpeg -init_hw_device vulkan=vk:0 -hwaccel vulkan -hwaccel_output_format vulkan -i ../video/A001_06091006_C002.mp4 -vf fps=6,hwdownload,format=nv12 -fps_mode vfr -q:v 2 ./frames/output_%05d.png
