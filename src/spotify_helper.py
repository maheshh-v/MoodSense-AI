
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os
import random

SPOTIPY_CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID")
SPOTIPY_CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET")

auth_manager = SpotifyClientCredentials(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET)
sp = spotipy.Spotify(auth_manager=auth_manager)


def get_top_songs_for_mood(mood):
    try:
        # Fetch more results for variety
        results = sp.search(q=mood + " music", type="track", limit=15)
        print(f"Spotify results for mood: {mood}")

        all_tracks = results["tracks"]["items"]
        if not all_tracks:
            raise Exception("No songs found for mood.")

        # Randomly pick 3 
        selected_tracks = random.sample(all_tracks, k=min(3, len(all_tracks)))

        songs = []
        for item in selected_tracks:
            title = item["name"] + " - " + item["artists"][0]["name"]
            url = item["external_urls"]["spotify"]
            print(f"Fetched song: {title} → {url}")
            songs.append({"title": title, "url": url})

        return songs

    except Exception as e:
        print("Spotify API Error:", e)
        return [{"title": "No Song Found", "url": "https://open.spotify.com/"}]
