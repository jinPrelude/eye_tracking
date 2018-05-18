# Eye-tracking
The goal of this project is making eye-tracking mouse-control software using ConvNet and OpenCV. Each descriptions of the codes in this project is below.

## descriptions
### record_makeDataset
- This code will record the videos via your webcam, And allow you to choose whether to draw rectangles on the images(which is separated by a frame), or leave it for a while and do using 'only_drawing.py'. By doing so, you can generate the dataset of the eyes-position.<br>
- Dataset will be stored in dataset folder, positions of the rectangles will be saved in 'eyesPos.csv', and the last number that you've generated rectangles' position will be saved in 'last_num.txt'
<div align="center">
<img style="display:inline;" src=gif/recording.gif width="245px"/>
<img style="display:inline;" src=gif/drawing.gif width="245px"/>
</div>
<br>
<br>


### only_drawing
-  You can draw rectangles using this program although you chose not to draw by pressing 'n' button in the record_makeDataset program so you chose not to draw rectangles right now. Code will automatically get the checkpoint through reading 'last_num.txt
