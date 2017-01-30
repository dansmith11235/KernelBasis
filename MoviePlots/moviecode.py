from imdbpie import Imdb # IMDB API Library
import re # Regular Expression Library
import numpy as np 
from wordcloud import WordCloud
from PIL import Image, ImageFont, ImageDraw
import matplotlib.pyplot as plt
import os # for the directory

#### Set up a connection to IMDb API
imdb = Imdb(anonymize=True) # to proxy requests

### Pull titles of top movies
titles = imdb.top_250()

#### Create empty list to store all the plots
TopPlots = []
#### loop through all 50 films
#### Need two loops because IMDBs API needs a break 
for i in range(0,25):
              
    #### Get The plots of each movie adds it to the list of plots 
    TopPlots = TopPlots + imdb.get_title_by_id(titles[i].get('tconst')).plots

            
for i in range(25,51):
                            
    #### Get The plots of each movie adds it to the list of plots 
    TopPlots = TopPlots + imdb.get_title_by_id(titles[i].get('tconst')).plots            

#### convert the list plots to one string
TopPlots = ''.join(TopPlots)

###Import the image of the oscar trophy
oscar = np.array(Image.open("oscar2.png"))

#### Generate a word cloud image
wordcloud = WordCloud(background_color="white", max_words=2000, mask=oscar).generate(TopPlots)

#### Display the generated image:
plt.imshow(wordcloud)
plt.axis("off")


plt.show()

