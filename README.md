# Extract_emotions_and_english_speaking_fluency_from_video
AI Based emotions and English fluency extraction  
# Requirements:

fer==21.0.5

ffmpeg==1.4

moviepy==1.0.3

mtcnn==0.1.1

tensorflow==2.7.0

SoundFile==0.10.3.post1

librosa==0.8.1

plotly==5.4.0

# Extract emotions:
Run emotion.py file and that will be extract emotions from the each images that you have specify in save_fps parameter..
output will be save in png file and vizualize as below..
![my](https://user-images.githubusercontent.com/65647192/146373238-a4eef860-a084-4615-9fbe-7a182d7803d3.png)
![ezgif com-gif-maker](https://user-images.githubusercontent.com/65647192/146373727-a8b28ebc-7951-414b-a98c-1137838ce755.gif)

# For English Fluency:
Run fluency.py file and that will extract audio from video that you given as input and further that audio will pass for classification and based on audio feature it classify into Beginner, Intermediate, Advanced Level of english speaking fluency..( model accuracy is 89% )

Audio Features:
![download](https://user-images.githubusercontent.com/65647192/146375652-a48251b0-a4d8-4213-8573-69171992f30e.png)


Output will be save in png file and vizualize as below..

![a](https://user-images.githubusercontent.com/65647192/146376679-b9b3acb0-3cae-4555-a73c-fde1b178f52d.png)
![b](https://user-images.githubusercontent.com/65647192/146376681-7d6ccd36-c6e1-4d7e-9b0a-23fef4d16a8a.png)
![c](https://user-images.githubusercontent.com/65647192/146376685-5be99920-1429-453b-af0f-0d344aa9dee7.png)

Refences:https://github.com/justinshenk/fer  https://github.com/krishnaik06/Audio-Classification
