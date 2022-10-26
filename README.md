# Fast-face-and-car-license-plate-recognition
This project is based on YOLOv5 for license plate and face recognition, a large amount of data training, and parallel and tensorrt acceleration, which can achieve a speed of more than 10FPS on jetson nano.

First, you need to install the corresponding environment according to the instructions in env.txt, and then run detect.py to get the visual recognition result.

<img src="https://github.com/pzyqwe/Fast-face-and-car-license-plate-recognition/blob/main/carface/output/1.jpg" width="950px">
<img src="https://github.com/pzyqwe/Fast-face-and-car-license-plate-recognition/blob/main/carface/output/face.png" width="950px">
<img src="https://github.com/pzyqwe/Fast-face-and-car-license-plate-recognition/blob/main/carface/output/car.png" width="950px">

|   Backbone   | Speed V100 F32(ms) | TensorRT_FP16(ms) |
| :----------: | :---------:        | :---------------: |
|  carface     |       0.9          |       0.32        |


