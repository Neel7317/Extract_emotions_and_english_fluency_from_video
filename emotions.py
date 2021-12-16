from fer import Video
from fer import FER
import matplotlib.pyplot as plt
import math,re,sys,os
from collections import Counter
#import moviepy.editor as mp

def aa():
    videofile = "whatsapp.mp4"
    # Face detection
    detector = FER(mtcnn=True)
    # Video predictions
    video = Video(videofile)

    # Output list of dictionaries
    raw_data = video.analyze(detector, display=False,save_fps=3)

    # Convert to pandas for analysis
    df = video.to_pandas(raw_data)
    df = video.get_first_face(df)
    df = video.get_emotions(df)

    per=[]
    for i in df.columns:
        per.append(df[i].sum()/(df.sum().sum())*100)

    
        
    label=['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    #mycolors=['red','pink','orange','cyan','lime','yellow','gray']  
    #plt.figure(figsize =(10, 7))
    #plt.pie(per, labels = label,colors=mycolors,autopct='%1.1f%%', radius= 1.0,startangle=90,textprops ={'fontsize':13}, pctdistance =0.70 )
    #Creating the donut shape for the pie
    #centre_circle = plt.Circle((0,0), 0.45, fc='white')
    #fig= plt.gcf()
    #fig.gca().add_artist(centre_circle)
    #plt.legend()
    plt.barh(label,per,color=['red','pink','orange','cyan','lime','yellow','gray'])
    plt.title('Emotion Chart')
    plt.ylabel('')
    plt.xlabel('Emotions %..')
    plt.savefig('my2.png')
    

aa()
