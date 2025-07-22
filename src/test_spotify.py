from dotenv import load_dotenv
import os
load_dotenv()  

from spotify_helper import get_top_songs_for_mood

print(get_top_songs_for_mood("love"))
