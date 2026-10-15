"""
Movie Recommendation System Using Natural Language Processing and Python
========================================================================
A modern, responsive Streamlit web application that recommends movies based
on content similarity (genres and descriptions) using TF-IDF Vectorization
and Cosine Similarity.

Author: pair programming with Antigravity
Dataset: movies.csv
"""

import os
import re
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------------------------------------------------------
# 1. Page Configuration & Modern Light Theme CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a modern, crisp light theme with pure white backgrounds
CUSTOM_CSS = """
<style>
    /* Global Background and Typography */
    .stApp {
        background-color: #FFFFFF !important;
        color: #1E293B !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Main Container Bounds */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3.5rem;
        max-width: 1240px;
    }

    /* Header Container */
    .header-box {
        text-align: center;
        padding: 2.25rem 1.5rem 1.75rem 1.5rem;
        margin-bottom: 1.75rem;
        border: 1px solid #E2E8F0;
        background: linear-gradient(180deg, #F8FAFC 0%, #FFFFFF 100%);
        border-radius: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04);
    }
    .header-title {
        font-size: 2.6rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.025em;
        margin-bottom: 0.35rem;
    }
    .header-subtitle {
        font-size: 1.2rem;
        font-weight: 500;
        color: #475569;
        margin-top: 0;
        margin-bottom: 0.75rem;
    }
    .tech-pill {
        display: inline-block;
        padding: 0.25rem 0.85rem;
        font-size: 0.825rem;
        font-weight: 600;
        border-radius: 9999px;
        background-color: #EFF6FF;
        color: #2563EB;
        border: 1px solid #BFDBFE;
    }

    /* Selected Movie Hero Card */
    .hero-card {
        background-color: #F8FAFC;
        border: 1px solid #CBD5E1;
        border-left: 6px solid #2563EB;
        border-radius: 14px;
        padding: 1.5rem 1.75rem;
        margin: 1.5rem 0 2rem 0;
        box-shadow: 0 4px 10px rgba(15, 23, 42, 0.05);
    }
    .hero-title {
        font-size: 1.75rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 0.4rem;
    }
    .hero-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 0.85rem;
        align-items: center;
        margin-bottom: 0.85rem;
    }

    /* Movie Recommendation Cards */
    .movie-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 1rem;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    }
    .movie-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 24px rgba(37, 99, 235, 0.12);
        border-color: #93C5FD;
    }

    .card-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }
    .card-rank {
        display: inline-block;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        padding: 0.2rem 0.55rem;
        border-radius: 6px;
        background-color: #DBEAFE;
        color: #1D4ED8;
    }
    .card-match {
        font-size: 0.8rem;
        font-weight: 700;
        color: #059669;
        background-color: #ECFDF5;
        padding: 0.2rem 0.55rem;
        border-radius: 6px;
        border: 1px solid #A7F3D0;
    }
    .card-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 0.4rem;
        margin-bottom: 0.25rem;
        line-height: 1.3;
    }
    .card-rating {
        font-size: 0.88rem;
        font-weight: 600;
        color: #D97706;
        margin-bottom: 0.5rem;
    }
    .card-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 0.35rem;
        margin-bottom: 0.65rem;
    }
    .genre-tag {
        font-size: 0.72rem;
        background-color: #F1F5F9;
        color: #334155;
        padding: 0.2rem 0.5rem;
        border-radius: 5px;
        font-weight: 500;
        border: 1px solid #E2E8F0;
    }
    .card-desc {
        font-size: 0.84rem;
        color: #64748B;
        line-height: 1.42;
        display: -webkit-box;
        -webkit-line-clamp: 4;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }

    /* Poster image styling */
    div[data-testid="stImage"] img {
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.08);
        object-fit: cover;
    }

    /* Sidebar Styling */
    div[data-testid="stSidebar"] {
        background-color: #F8FAFC !important;
        border-right: 1px solid #E2E8F0;
    }

    /* Buttons styling */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.15s ease-in-out;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. Dataset Management & Preprocessing
# -----------------------------------------------------------------------------
FALLBACK_DATA = [
    {
        "movie_id": 1,
        "title": "Inception",
        "genre": "Sci-Fi Action Thriller",
        "rating": 8.8,
        "release_year": 2010,
        "description": "A skilled thief who steals corporate secrets through dream-sharing technology is given the task of planting an idea into the mind of a CEO with a team of specialists.",
        "poster_url": "https://image.tmdb.org/t/p/w500/edv5CZvWj09upOsy2Y6IwDhK8bt.jpg"
    },
    {
        "movie_id": 2,
        "title": "Interstellar",
        "genre": "Sci-Fi Adventure Drama",
        "rating": 8.7,
        "release_year": 2014,
        "description": "A team of exploratory astronauts and scientists travel through a wormhole near Saturn in search of a new habitable planet for mankind as Earth collapses.",
        "poster_url": "https://image.tmdb.org/t/p/w500/gEU2QniE6E77NI6lCU6MxlNBvIx.jpg"
    },
    {
        "movie_id": 3,
        "title": "The Dark Knight",
        "genre": "Action Crime Drama",
        "rating": 9.0,
        "release_year": 2008,
        "description": "Batman raises the stakes in his war on crime to dismantle organized crime in Gotham City, only to face a sadistic criminal mastermind known as the Joker.",
        "poster_url": "https://image.tmdb.org/t/p/w500/qJ2tW6WMUDux911r6m7haRef0WH.jpg"
    },
    {
        "movie_id": 4,
        "title": "The Matrix",
        "genre": "Sci-Fi Action Cyberpunk",
        "rating": 8.7,
        "release_year": 1999,
        "description": "A computer hacker named Neo learns from mysterious rebels about the true nature of his reality and his role in the war against machine controllers of a simulated dream world.",
        "poster_url": "https://image.tmdb.org/t/p/w500/f89U3ADr1oiB1s9GkdPOEpXUk5H.jpg"
    },
    {
        "movie_id": 5,
        "title": "Pulp Fiction",
        "genre": "Crime Drama Dark-Comedy",
        "rating": 8.9,
        "release_year": 1994,
        "description": "The lives of two mob hitmen, a boxer, a gangster and his wife, and a pair of diner bandits intertwine in four tales of violence, redemption, and eccentric dialogue.",
        "poster_url": "https://image.tmdb.org/t/p/w500/d5iIlFn5s0ImszYzBPb8JPIfbXD.jpg"
    },
    {
        "movie_id": 6,
        "title": "The Shawshank Redemption",
        "genre": "Drama Crime",
        "rating": 9.3,
        "release_year": 1994,
        "description": "A banker convicted of murder spends two decades at Shawshank State Penitentiary, finding solace and redemption through friendship with fellow inmate Red.",
        "poster_url": "https://image.tmdb.org/t/p/w500/9cqNxx0GxF0bflZmeSMuL5tnGzr.jpg"
    },
    {
        "movie_id": 7,
        "title": "The Godfather",
        "genre": "Crime Drama Mafia",
        "rating": 9.2,
        "release_year": 1972,
        "description": "The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant and youngest son Michael Corleone.",
        "poster_url": "https://image.tmdb.org/t/p/w500/3bhkrj58Vtu7enYsRolD1fZdja1.jpg"
    },
    {
        "movie_id": 8,
        "title": "Gladiator",
        "genre": "Action Adventure Drama",
        "rating": 8.5,
        "release_year": 2000,
        "description": "A former Roman general sets out to exact vengeance against the corrupt prince who murdered his family and sent him into slavery as a gladiator in the Colosseum.",
        "poster_url": "https://image.tmdb.org/t/p/w500/ty8TGRuvJLPUmAR1H1nRIsgwvim.jpg"
    },
    {
        "movie_id": 9,
        "title": "Fight Club",
        "genre": "Drama Psychological Thriller",
        "rating": 8.8,
        "release_year": 1999,
        "description": "An insomniac office worker looking for a way to change his life forms an underground fight club that evolves into something much more dangerous.",
        "poster_url": "https://image.tmdb.org/t/p/w500/pB8BM7pdSp6B6Ih7QZ4DrQ3PmJK.jpg"
    },
    {
        "movie_id": 10,
        "title": "Forrest Gump",
        "genre": "Drama Romance Comedy",
        "rating": 8.8,
        "release_year": 1994,
        "description": "The history of the United States unfolds through the perspective of an Alabama man with an IQ of 75 who yearns to be reunited with his childhood sweetheart.",
        "poster_url": "https://image.tmdb.org/t/p/w500/arw2vcBveWOVZr6pxd9XTd1TdQa.jpg"
    },
    {
        "movie_id": 11,
        "title": "Avengers: Endgame",
        "genre": "Action Sci-Fi Superhero",
        "rating": 8.4,
        "release_year": 2019,
        "description": "After devastating cosmic events, the remaining Avengers assemble once more to reverse Thanos' universe-shattering actions and restore balance.",
        "poster_url": "https://image.tmdb.org/t/p/w500/or06FN3Dka5tukK1e9sl16pB3iy.jpg"
    },
    {
        "movie_id": 12,
        "title": "Spider-Man: Into the Spider-Verse",
        "genre": "Animation Action Adventure",
        "rating": 8.4,
        "release_year": 2018,
        "description": "Teenager Miles Morales becomes the new Spider-Man and joins spider-powered heroes from other dimensions to stop a threat to all realities.",
        "poster_url": "https://image.tmdb.org/t/p/w500/iiZZdoQBEYBv6id8su7ImL0oCbD.jpg"
    },
    {
        "movie_id": 13,
        "title": "Blade Runner 2049",
        "genre": "Sci-Fi Mystery Thriller",
        "rating": 8.0,
        "release_year": 2017,
        "description": "Young Blade Runner K uncovers a long-buried secret that leads him to track down former Blade Runner Rick Deckard in a dystopian cybernetic future.",
        "poster_url": "https://image.tmdb.org/t/p/w500/gajva2L0rPYkEWjzgFlBXCAVBE5.jpg"
    },
    {
        "movie_id": 14,
        "title": "Arrival",
        "genre": "Sci-Fi Drama Mystery",
        "rating": 7.9,
        "release_year": 2016,
        "description": "A linguist is recruited by the military to communicate with extraterrestrial visitors after twelve mysterious spacecraft appear around the globe.",
        "poster_url": "https://image.tmdb.org/t/p/w500/x2FJsf1ElAgr63Y3PNPtJrcmpoe.jpg"
    },
    {
        "movie_id": 15,
        "title": "Tenet",
        "genre": "Sci-Fi Action Thriller",
        "rating": 7.3,
        "release_year": 2020,
        "description": "Armed with only one word, a secret agent journeys through international espionage on a mission that unfolds beyond real-time physics and entropy.",
        "poster_url": "https://image.tmdb.org/t/p/w500/aKx1ARwG55zZ0GpRvU2WrGrCG9P.jpg"
    },
    {
        "movie_id": 16,
        "title": "The Silence of the Lambs",
        "genre": "Crime Drama Thriller",
        "rating": 8.6,
        "release_year": 1991,
        "description": "A young FBI cadet receives help from an incarcerated cannibalistic psychiatrist to track down an elusive serial killer targeting women.",
        "poster_url": "https://image.tmdb.org/t/p/w500/uS9m8OBk1A8eM9I042bx8XXpqAq.jpg"
    },
    {
        "movie_id": 17,
        "title": "Se7en",
        "genre": "Crime Mystery Drama",
        "rating": 8.6,
        "release_year": 1995,
        "description": "Two detectives, a rookie and a veteran, hunt a serial killer who uses the seven deadly sins as his motives in a grim, rainy metropolis.",
        "poster_url": "https://image.tmdb.org/t/p/w500/6yoghtyTpznpBik8EngEmJskVUO.jpg"
    },
    {
        "movie_id": 18,
        "title": "Shutter Island",
        "genre": "Mystery Thriller Psychological",
        "rating": 8.2,
        "release_year": 2010,
        "description": "A U.S. Marshal investigates the disappearance of a patient who escaped from a hospital for the criminally insane on a desolate Boston island.",
        "poster_url": "https://image.tmdb.org/t/p/w500/kve20wg7UT4vX508fAP5q1gW1pL.jpg"
    },
    {
        "movie_id": 19,
        "title": "Spirited Away",
        "genre": "Animation Adventure Fantasy",
        "rating": 8.6,
        "release_year": 2001,
        "description": "A sullen 10-year-old girl wanders into an enchanting world ruled by gods, witches, and spirits, where humans are transformed into beasts.",
        "poster_url": "https://image.tmdb.org/t/p/w500/39wmItIWsg5sZMyRUHLkWBcuVCM.jpg"
    },
    {
        "movie_id": 20,
        "title": "The Lion King",
        "genre": "Animation Adventure Drama",
        "rating": 8.5,
        "release_year": 1994,
        "description": "Lion cub prince Simba is targeted by his bitter uncle Scar. Following tragic exile, Simba must find courage to reclaim the Pride Lands.",
        "poster_url": "https://image.tmdb.org/t/p/w500/sKCr78MXSLixwmZ8DyJLrpMsd15.jpg"
    },
    {
        "movie_id": 21,
        "title": "Coco",
        "genre": "Animation Adventure Family",
        "rating": 8.4,
        "release_year": 2017,
        "description": "Aspiring musician Miguel enters the vibrant Land of the Dead to unlock the real story behind his family's ancestral ban on music.",
        "poster_url": "https://image.tmdb.org/t/p/w500/gGEsBPAijhVUFoiNpgZXqRVWJt2.jpg"
    },
    {
        "movie_id": 22,
        "title": "WALL-E",
        "genre": "Animation Adventure Sci-Fi",
        "rating": 8.4,
        "release_year": 2008,
        "description": "In the distant future, a small waste-collecting robot inadvertently embarks on a cosmic journey that determines the ultimate fate of humanity.",
        "poster_url": "https://image.tmdb.org/t/p/w500/hbhFnRzzg6ZDmm8YAmxBnQpQIPh.jpg"
    },
    {
        "movie_id": 23,
        "title": "Toy Story",
        "genre": "Animation Comedy Family",
        "rating": 8.3,
        "release_year": 1995,
        "description": "A classic cowboy doll is profoundly threatened and jealous when a fancy new space ranger action figure supplants him in a child's bedroom.",
        "poster_url": "https://image.tmdb.org/t/p/w500/uXDfjJbdP4ijW5hWSBrPrlKpxab.jpg"
    },
    {
        "movie_id": 24,
        "title": "Finding Nemo",
        "genre": "Animation Adventure Comedy",
        "rating": 8.2,
        "release_year": 2003,
        "description": "After his son is captured in the Great Barrier Reef and taken to Sydney, a timid clownfish sets out on a daring ocean journey to bring him home.",
        "poster_url": "https://image.tmdb.org/t/p/w500/eHuGQ10FUzK1mdOY692F5qG9lMw.jpg"
    },
    {
        "movie_id": 25,
        "title": "Up",
        "genre": "Animation Adventure Comedy",
        "rating": 8.3,
        "release_year": 2009,
        "description": "An elderly widower travels to Paradise Falls in his balloon-powered flying house, inadvertently taking an eager young wilderness explorer along.",
        "poster_url": "https://image.tmdb.org/t/p/w500/vpbaStTMt8qqHaEvebt2UOYgdIY.jpg"
    },
    {
        "movie_id": 26,
        "title": "The Lord of the Rings: The Fellowship of the Ring",
        "genre": "Adventure Fantasy Action",
        "rating": 8.8,
        "release_year": 2001,
        "description": "A meek Hobbit and eight heroic companions embark on an epic quest to destroy the One Ring and save Middle-earth from the Dark Lord Sauron.",
        "poster_url": "https://image.tmdb.org/t/p/w500/6oom5QYQ2yQTMJIbnvbkBL9cDK6.jpg"
    },
    {
        "movie_id": 27,
        "title": "The Lord of the Rings: The Return of the King",
        "genre": "Adventure Fantasy Action",
        "rating": 9.0,
        "release_year": 2003,
        "description": "Gandalf and Aragorn lead the armies of Men in a monumental final stand to divert Sauron's gaze while Frodo and Sam approach Mount Doom.",
        "poster_url": "https://image.tmdb.org/t/p/w500/rCzpDGLbOoPwLjy3OAm5NUPOTrC.jpg"
    },
    {
        "movie_id": 28,
        "title": "Star Wars: A New Hope",
        "genre": "Sci-Fi Action Adventure",
        "rating": 8.6,
        "release_year": 1977,
        "description": "Luke Skywalker joins forces with a Jedi Knight, a cocky pilot, and two droids to rescue Princess Leia and destroy the Empire's Death Star.",
        "poster_url": "https://image.tmdb.org/t/p/w500/6FfCtAuVAW8XJjZ7eWeLibRLWTw.jpg"
    },
    {
        "movie_id": 29,
        "title": "Jurassic Park",
        "genre": "Sci-Fi Adventure Thriller",
        "rating": 8.2,
        "release_year": 1993,
        "description": "A paleontologist touring a secluded theme park populated by cloned dinosaurs must fight to survive when a security sabotage unleashes predators.",
        "poster_url": "https://image.tmdb.org/t/p/w500/oU7Oq2kFAAlGqbU4VoAE36g4hoI.jpg"
    },
    {
        "movie_id": 30,
        "title": "Avatar",
        "genre": "Sci-Fi Action Adventure",
        "rating": 7.9,
        "release_year": 2009,
        "description": "A paraplegic Marine dispatched to the alien moon Pandora becomes torn between following military orders and protecting the indigenous Na'vi.",
        "poster_url": "https://image.tmdb.org/t/p/w500/kyeqWdyUXW608qlYkRqosgbbnKR.jpg"
    },
    {
        "movie_id": 31,
        "title": "Titanic",
        "genre": "Romance Drama History",
        "rating": 7.9,
        "release_year": 1997,
        "description": "A seventeen-year-old aristocrat falls deeply in love with a kind-hearted but penniless artist aboard the ill-fated R.M.S. Titanic.",
        "poster_url": "https://image.tmdb.org/t/p/w500/9xjZS2rlVxm8SFx8kPC3aIGCOYQ.jpg"
    },
    {
        "movie_id": 32,
        "title": "La La Land",
        "genre": "Comedy Drama Music Romance",
        "rating": 8.0,
        "release_year": 2016,
        "description": "While navigating their careers in Los Angeles, a passionate jazz pianist and an aspiring actress fall in love while chasing their artistic dreams.",
        "poster_url": "https://image.tmdb.org/t/p/w500/uDO8zWDhfWwoFdKS4fzkVJb0Rf0.jpg"
    },
    {
        "movie_id": 33,
        "title": "Eternal Sunshine of the Spotless Mind",
        "genre": "Romance Sci-Fi Drama",
        "rating": 8.3,
        "release_year": 2004,
        "description": "When a passionate relationship turns bitter, a couple undergoes a clinical memory-erasure procedure, only to discover their love transcends memory.",
        "poster_url": "https://image.tmdb.org/t/p/w500/5MwkWH9tYHv3mV9OdYTMR5qreIz.jpg"
    },
    {
        "movie_id": 34,
        "title": "Goodfellas",
        "genre": "Biography Crime Drama",
        "rating": 8.7,
        "release_year": 1990,
        "description": "The rise and fall of mob associate Henry Hill as he climbs through the ranks of the New York Italian-American crime syndicate.",
        "poster_url": "https://image.tmdb.org/t/p/w500/aKuFiU82s5ISJpGZp7YkIr3kcu.jpg"
    },
    {
        "movie_id": 35,
        "title": "The Departed",
        "genre": "Crime Drama Thriller",
        "rating": 8.5,
        "release_year": 2006,
        "description": "An undercover state trooper and an informant in the Massachusetts State Police desperately race to uncover each other's secret identities.",
        "poster_url": "https://image.tmdb.org/t/p/w500/nT97ifVT2J1yMQmeq20Qblg61T.jpg"
    },
    {
        "movie_id": 36,
        "title": "Mad Max: Fury Road",
        "genre": "Action Adventure Sci-Fi",
        "rating": 8.1,
        "release_year": 2015,
        "description": "In a post-apocalyptic desert wasteland, Imperator Furiosa and drifter Max Rockatansky lead a high-octane rebellion against an oppressive warlord.",
        "poster_url": "https://image.tmdb.org/t/p/w500/8tZYtuWezp8JbcsvHYO0O46tFbo.jpg"
    },
    {
        "movie_id": 37,
        "title": "John Wick",
        "genre": "Action Crime Thriller",
        "rating": 7.4,
        "release_year": 2014,
        "description": "A retired elite assassin is provoked back into action after Russian mobsters steal his vintage Mustang and kill the puppy gifted by his late wife.",
        "poster_url": "https://image.tmdb.org/t/p/w500/fZPSd91yGE9fCcCe6OoQr6E3Bev.jpg"
    },
    {
        "movie_id": 38,
        "title": "Parasite",
        "genre": "Drama Thriller Comedy",
        "rating": 8.5,
        "release_year": 2019,
        "description": "Greed and class discrimination ignite an unexpected conflict when a destitute family schemes to become employed by an affluent household.",
        "poster_url": "https://image.tmdb.org/t/p/w500/7IiTTgloJzvGI1TAYymCfbfl3vT.jpg"
    },
    {
        "movie_id": 39,
        "title": "A Quiet Place",
        "genre": "Horror Sci-Fi Drama",
        "rating": 7.5,
        "release_year": 2018,
        "description": "In a post-apocalyptic world infested with blind predatory monsters that hunt by sound, a courageous family must navigate their lives in absolute silence.",
        "poster_url": "https://image.tmdb.org/t/p/w500/nAU74GmpUk7t5iklEp3bufwDq4n.jpg"
    },
    {
        "movie_id": 40,
        "title": "Get Out",
        "genre": "Horror Mystery Thriller",
        "rating": 7.8,
        "release_year": 2017,
        "description": "A young African-American visits his white girlfriend's secluded family estate for the weekend, uncovering a disturbing psychological conspiracy.",
        "poster_url": "https://image.tmdb.org/t/p/w500/tFXcEccSQMf3lfhfXKSU9iRBpa3.jpg"
    },
]

DEFAULT_POSTER_PLACEHOLDER = "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=500&auto=format&fit=crop&q=80"


def clean_text(text: str) -> str:
    """Basic NLP Text Preprocessing:

    - Convert to lowercase
    - Strip punctuation and non-alphanumeric characters
    - Normalize whitespace
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@st.cache_data(show_spinner=False)
def load_and_preprocess_data(dataset_path: str = "movies.csv"):
    """Load dataset, clean textual features, construct TF-IDF numerical matrix,

    and calculate Cosine Similarity matrix.
    """
    # 1. Load CSV dataset or fallback to sample
    if os.path.exists(dataset_path):
        try:
            df = pd.read_csv(dataset_path)
            required_cols = {"title", "genre", "description"}
            if not required_cols.issubset(df.columns):
                df = pd.DataFrame(FALLBACK_DATA)
        except Exception:
            df = pd.DataFrame(FALLBACK_DATA)
    else:
        df = pd.DataFrame(FALLBACK_DATA)
        try:
            df.to_csv(dataset_path, index=False)
        except Exception:
            pass

    # 2. Handle missing data gracefully
    for col in ["title", "genre", "description"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    if "rating" not in df.columns:
        df["rating"] = 8.0
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(7.5)

    if "release_year" not in df.columns:
        df["release_year"] = "N/A"

    if "poster_url" not in df.columns:
        df["poster_url"] = DEFAULT_POSTER_PLACEHOLDER
    else:
        df["poster_url"] = df["poster_url"].fillna(DEFAULT_POSTER_PLACEHOLDER).replace("", DEFAULT_POSTER_PLACEHOLDER)

    # 3. Clean textual features using basic NLP
    df["clean_genre"] = df["genre"].apply(clean_text)
    df["clean_description"] = df["description"].apply(clean_text)

    # 4. Feature Combination: duplicate genre to give appropriate weight to movie category
    df["combined_features"] = (
        df["clean_genre"]
        + " "
        + df["clean_genre"]
        + " "
        + df["clean_description"]
    )

    # 5. TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),  # unigrams and bigrams for context
        max_features=5000,
    )
    tfidf_matrix = tfidf_vectorizer.fit_transform(df["combined_features"])

    # 6. Cosine Similarity Calculation
    cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return df, cosine_sim_matrix, tfidf_vectorizer


def recommend_movies(
    movie_title: str, df: pd.DataFrame, similarity_matrix: np.ndarray, top_n: int = 5
):
    """Recommend top N similar movies for a given movie title, excluding itself."""
    if not movie_title or df.empty:
        return None, []

    # Locate movie index (exact match or case-insensitive)
    matches = df[df["title"].str.strip().str.lower() == movie_title.strip().lower()]

    if matches.empty:
        # Substring / partial match fallback
        matches = df[
            df["title"]
            .str.strip()
            .str.lower()
            .str.contains(re.escape(movie_title.strip().lower()))
        ]
        if matches.empty:
            return None, []

    movie_idx = matches.index[0]
    selected_movie_data = df.iloc[movie_idx].to_dict()

    # Extract similarity scores
    sim_scores = list(enumerate(similarity_matrix[movie_idx]))

    # Sort descending by similarity
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Exclude the selected movie itself
    filtered_scores = [item for item in sim_scores if item[0] != movie_idx]

    # Select top N recommendations
    top_indices_scores = filtered_scores[:top_n]

    recommendations = []
    # Highest similarity score among recommendations for calibration
    max_score = top_indices_scores[0][1] if top_indices_scores else 1.0

    for idx, score in top_indices_scores:
        movie_row = df.iloc[idx].to_dict()
        raw_score = float(score)
        movie_row["similarity_score"] = raw_score
        # Calibrated relative match percentage for intuitive display
        calibrated_pct = int(round(65 + (raw_score / max(max_score, 0.001)) * 33))
        movie_row["match_percent"] = min(99, max(60, calibrated_pct))
        recommendations.append(movie_row)

    return selected_movie_data, recommendations


# -----------------------------------------------------------------------------
# 3. Main Streamlit Interface
# -----------------------------------------------------------------------------
def main():
    # Load dataset and NLP vectors
    df, cosine_sim, _ = load_and_preprocess_data("movies.csv")
    all_movies = sorted(df["title"].unique().tolist())

    # State management
    if "search_input" not in st.session_state:
        st.session_state.search_input = ""
    if "selected_movie" not in st.session_state:
        st.session_state.selected_movie = all_movies[0] if all_movies else ""
    if "show_recommendations" not in st.session_state:
        st.session_state.show_recommendations = False

    # Header section
    st.markdown(
        """
        <div class="header-box">
            <div class="header-title">🎬 Movie Recommendation System</div>
            <div class="header-subtitle">Discover your next favorite movie!</div>
            <div class="tech-pill">Natural Language Processing • TF-IDF Vectorization • Cosine Similarity</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # Sidebar
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.subheader("💡 NLP & Recommendation Pipeline")
        st.markdown(
            """
            This application implements **Content-Based Filtering**:
            
            1. **Text Preprocessing**: Cleans movie genres and overviews using regex tokenization and lowercasing.
            2. **TF-IDF Vectorization**: Transforms textual descriptions into high-dimensional vector representations.
            3. **Cosine Similarity**: Measures angular similarity between movie feature vectors:
            $$\\text{Cosine Similarity}(A, B) = \\frac{A \\cdot B}{\\|A\\| \\|B\\|}$$
            """
        )

        st.markdown("---")
        st.subheader("📊 Dataset Statistics")
        c1, c2 = st.columns(2)
        c1.metric("Total Movies", len(df))
        avg_rating = df["rating"].mean() if not df.empty else 0
        c2.metric("Avg Rating", f"{avg_rating:.1f} ⭐")

        st.markdown("---")
        st.subheader("⚡ Quick Try Samples")
        st.write("Click any title to load recommendations instantly:")
        quick_picks = ["Inception", "The Godfather", "Toy Story", "The Dark Knight", "Titanic"]
        for qp in quick_picks:
            if qp in all_movies:
                if st.button(f"🎥 {qp}", use_container_width=True, key=f"quick_{qp}"):
                    st.session_state.selected_movie = qp
                    st.session_state.search_input = ""
                    st.session_state.show_recommendations = True
                    st.rerun()

        st.markdown("---")
        st.caption("Developed with Streamlit, Scikit-learn, Pandas & NumPy.")

    # -------------------------------------------------------------------------
    # Search Box, Dropdown, and Action Controls
    # -------------------------------------------------------------------------
    st.write("### 🔍 Find a Movie")

    # Row 1: Search Text Filter
    search_col1, search_col2 = st.columns([4, 1])
    with search_col1:
        search_query = st.text_input(
            "Type movie title or keyword to filter options:",
            value=st.session_state.search_input,
            placeholder="e.g. Inception, Godfather, Crime, Space, Animation...",
            help="Type keywords to filter the dropdown list below.",
        )
        if search_query != st.session_state.search_input:
            st.session_state.search_input = search_query

    with search_col2:
        st.write("")
        st.write("")
        if st.button("❌ Clear Search", use_container_width=True):
            st.session_state.search_input = ""
            st.rerun()

    # Filter dropdown options if search query is provided
    if st.session_state.search_input.strip():
        term = st.session_state.search_input.strip().lower()
        filtered_movies = [
            m for m in all_movies
            if term in m.lower() or term in df[df["title"] == m]["genre"].values[0].lower()
        ]
        if not filtered_movies:
            st.warning(f"No movies found matching '{st.session_state.search_input}'. Showing full movie catalog.")
            dropdown_options = all_movies
        else:
            dropdown_options = filtered_movies
    else:
        dropdown_options = all_movies

    # Row 2: Dropdown Selection & Action Buttons
    row2_col1, row2_col2, row2_col3 = st.columns([3, 1.2, 1], gap="medium")

    with row2_col1:
        # Determine initial selection index
        default_idx = 0
        if st.session_state.selected_movie in dropdown_options:
            default_idx = dropdown_options.index(st.session_state.selected_movie)

        chosen_movie = st.selectbox(
            "Select a movie from the catalog:",
            options=dropdown_options,
            index=default_idx,
            help="Select the movie you'd like recommendations for.",
        )

    with row2_col2:
        st.write("")
        st.write("")
        recommend_btn = st.button("🔍 Recommend Movies", type="primary", use_container_width=True)

    with row2_col3:
        st.write("")
        st.write("")
        reset_btn = st.button("🔄 Reset All", type="secondary", use_container_width=True)

    # Handle Reset All
    if reset_btn:
        st.session_state.search_input = ""
        st.session_state.selected_movie = all_movies[0] if all_movies else ""
        st.session_state.show_recommendations = False
        st.rerun()

    # Handle Recommend Click
    if recommend_btn:
        st.session_state.selected_movie = chosen_movie
        st.session_state.show_recommendations = True

    # -------------------------------------------------------------------------
    # Display Selected Movie & Recommendations
    # -------------------------------------------------------------------------
    current_movie = st.session_state.selected_movie

    if st.session_state.show_recommendations and current_movie:
        selected_data, recommendations = recommend_movies(
            current_movie, df, cosine_sim, top_n=5
        )

        if not selected_data:
            st.error(f"Movie '{current_movie}' could not be resolved. Please choose another title.")
            return

        # 1. Selected Movie Hero Banner (with Poster & Metadata)
        st.markdown("---")
        st.subheader("🎯 Selected Movie Details")

        hero_col_img, hero_col_info = st.columns([1, 4], gap="large")
        with hero_col_img:
            poster_src = selected_data.get("poster_url") or DEFAULT_POSTER_PLACEHOLDER
            st.image(poster_src, use_container_width=True)

        with hero_col_info:
            genre_tags_html = "".join(
                [f"<span class='genre-tag'>{g}</span>" for g in selected_data.get("genre", "").split()]
            )
            yr = selected_data.get("release_year", "N/A")
            rt = selected_data.get("rating", "N/A")

            st.markdown(
                f"""
                <div class="hero-card" style="margin-top: 0;">
                    <div class="hero-title">{selected_data.get('title')}</div>
                    <div class="hero-meta">
                        <span style="font-weight: 700; color: #D97706; font-size: 1.1rem;">⭐ {rt} / 10</span>
                        <span style="color: #64748B; font-weight: 500;">• Year: {yr}</span>
                        <div style="display: inline-flex; gap: 4px; margin-left: 8px;">{genre_tags_html}</div>
                    </div>
                    <div style="color: #334155; line-height: 1.6; font-size: 0.98rem; margin-top: 0.5rem;">
                        <strong>Plot Overview:</strong><br>{selected_data.get('description')}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # 2. Top 5 Recommendations Cards (Posters, Titles, Badges, Ratings, Genres & Descriptions)
        st.markdown("---")
        st.subheader(f"✨ Top 5 Recommended Movies for Fans of '{selected_data.get('title')}'")

        if not recommendations:
            st.warning("Insufficient similar titles found in catalog.")
        else:
            cols = st.columns(len(recommendations), gap="medium")

            for rank, (col, rec) in enumerate(zip(cols, recommendations), start=1):
                with col:
                    rec_poster = rec.get("poster_url") or DEFAULT_POSTER_PLACEHOLDER
                    st.image(rec_poster, use_container_width=True)

                    rec_genres_html = "".join(
                        [f"<span class='genre-tag'>{g}</span>" for g in rec.get("genre", "").split()]
                    )
                    st.markdown(
                        f"""
                        <div class="movie-card">
                            <div>
                                <div class="card-top">
                                    <span class="card-rank">#{rank} Pick</span>
                                    <span class="card-match">{rec.get('match_percent')}% Match</span>
                                </div>
                                <div class="card-title">{rec.get('title')}</div>
                                <div class="card-rating">⭐ {rec.get('rating')} <span style="font-size:0.8rem; color:#64748B;">({rec.get('release_year')})</span></div>
                                <div class="card-tags">{rec_genres_html}</div>
                            </div>
                            <div class="card-desc" title="{rec.get('description')}">
                                {rec.get('description')}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # 3. Transparent Technical Breakdown
            with st.expander("📊 Technical Breakdown & Cosine Similarity Scores"):
                breakdown_list = []
                for i, r in enumerate(recommendations, 1):
                    breakdown_list.append({
                        "Rank": f"#{i}",
                        "Movie Title": r.get("title"),
                        "Genres": r.get("genre"),
                        "IMDb Rating": f"⭐ {r.get('rating')}",
                        "Release Year": r.get("release_year"),
                        "Cosine Similarity": f"{r.get('similarity_score', 0):.4f}",
                        "Match Score": f"{r.get('match_percent')}%"
                    })
                breakdown_df = pd.DataFrame(breakdown_list)
                st.dataframe(breakdown_df, use_container_width=True, hide_index=True)

    else:
        # Welcome Guide and Featured Preview Cards
        st.markdown("---")
        st.info("👆 **Get Started**: Select a movie above and click **'🔍 Recommend Movies'** or click any quick pick in the sidebar.")

        st.subheader("🍿 Featured Movies in Catalog")
        grid_cols = st.columns(3, gap="medium")
        sample_movies = df.head(6).to_dict("records")

        for idx, movie in enumerate(sample_movies):
            with grid_cols[idx % 3]:
                feat_poster = movie.get("poster_url") or DEFAULT_POSTER_PLACEHOLDER
                st.image(feat_poster, use_container_width=True)
                sample_tags = "".join(
                    [f"<span class='genre-tag'>{g}</span>" for g in movie.get("genre", "").split()[:3]]
                )
                st.markdown(
                    f"""
                    <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 1.15rem; margin-bottom: 1rem; box-shadow: 0 2px 6px rgba(0,0,0,0.03);">
                        <div style="font-weight: 700; font-size: 1.1rem; color: #0F172A;">{movie.get('title')}</div>
                        <div style="font-size: 0.85rem; color: #D97706; margin: 0.25rem 0 0.5rem 0;">⭐ {movie.get('rating')} • {movie.get('release_year')}</div>
                        <div style="display: flex; gap: 4px; margin-bottom: 0.6rem;">{sample_tags}</div>
                        <div style="font-size: 0.825rem; color: #64748B; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;">{movie.get('description')}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


if __name__ == "__main__":
    main()
