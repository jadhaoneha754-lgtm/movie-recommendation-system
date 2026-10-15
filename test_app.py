"""Test script to verify data loading, NLP preprocessing, poster URLs, and recommendation functionality."""
from app import load_and_preprocess_data, recommend_movies

def test_pipeline():
    print("Testing data loading and preprocessing...")
    df, sim, tfidf = load_and_preprocess_data("movies.csv")
    print(f"Total movies loaded: {len(df)}")
    assert len(df) >= 30, f"Expected at least 30 movies, got {len(df)}"
    assert "poster_url" in df.columns, "poster_url column missing from dataset!"

    test_titles = ["Inception", "The Godfather", "Toy Story", "Titanic"]
    for title in test_titles:
        print(f"\n==========================================")
        print(f"Testing recommendations for: '{title}'")
        selected, recs = recommend_movies(title, df, sim, top_n=5)
        assert selected is not None, f"Failed to find movie: {title}"
        assert len(recs) == 5, f"Expected 5 recommendations, got {len(recs)}"
        assert selected.get("poster_url"), f"Poster URL missing for {title}"
        print(f"Selected: {selected['title']} | Genre: {selected['genre']} | Rating: {selected['rating']}")
        print(f"Poster: {selected['poster_url'][:60]}...")
        print("Top 5 Recommendations:")
        for i, rec in enumerate(recs, 1):
            assert rec['title'] != selected['title'], "Selected movie should not be recommended to itself!"
            assert rec.get('poster_url'), f"Poster URL missing for recommendation {rec['title']}"
            print(f"  {i}. {rec['title']} (Rating: {rec['rating']}, Match: {rec['match_percent']}%) - Genre: {rec['genre']}")
    
    print("\nAll unit tests passed successfully!")

if __name__ == "__main__":
    test_pipeline()
