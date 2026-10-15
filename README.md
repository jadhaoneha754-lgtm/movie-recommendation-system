# 🎬 Movie Recommendation System Using Natural Language Processing and Python

A complete, beginner-friendly, and production-ready mini project implementing a **Content-Based Movie Recommendation System** powered by **Natural Language Processing (NLP)**, **TF-IDF Vectorization**, **Cosine Similarity**, and an interactive **Streamlit** web user interface.

---

## 📌 Project Overview

This project recommends movies based on the semantic similarity of their content (genres and plot descriptions). When a user selects or searches for any movie, the machine learning pipeline calculates the mathematical closeness between the selected movie's textual features and all other movies in the catalog, returning the **Top 5 most similar movies** alongside posters, match scores, ratings, genres, and release years.

---

## ✨ Features

- **Modern Clean Light UI**: Minimalist white theme (`#FFFFFF`) with soft shadows, responsive cards, and clean typography.
- **Movie Posters**: High-resolution cover artwork for all movies in both the hero selected view and the recommendation grid.
- **NLP Text Preprocessing**: Tokenization, lowercasing, regex cleaning, and stopword removal.
- **TF-IDF Vectorization**: Converts text features (genres + plot summaries) into high-dimensional numerical feature vectors with unigram and bigram representation.
- **Cosine Similarity Engine**: Computes angular similarity between vector embeddings to rank recommendations accurately.
- **Dual Movie Search**:
  - **Live Filter Search Box**: Type any keyword or title to filter options in real time.
  - **Dropdown Selector**: Browse or pick from the catalog with type-ahead search.
- **Controls & Action Buttons**:
  - `🔍 Recommend Movies`: Generates top 5 similar recommendations.
  - `❌ Clear Search`: Clears search filters instantly.
  - `🔄 Reset All`: Resets state and returns to catalog view.
  - `🎯 Quick Try`: Instant one-click recommendations for popular titles in the sidebar.
- **Selected Movie Showcase**: Hero card highlighting the selected movie's poster, title, IMDb rating, release year, genre tags, and plot overview.
- **Top 5 Recommendation Cards**: Exactly 5 recommendations featuring:
  - Movie poster image
  - Rank badge (`#1 Pick`, `#2 Pick`, etc.)
  - Match score percentage
  - IMDb star rating and release year
  - Categorical genre badges
  - Plot overview snippet
- **Transparent ML Insights**: Expandable breakdown table showing exact mathematical Cosine Similarity scores.
- **Robust & Self-Healing**: Automatically creates or repairs the dataset if missing or malformed.

---

## 🛠️ Technology Stack

| Technology | Purpose |
| :--- | :--- |
| **Python 3.10+** | Core programming language |
| **Streamlit** | Modern, responsive web interface |
| **Pandas** | Tabular data manipulation and CSV handling |
| **NumPy** | High-performance matrix calculations |
| **Scikit-learn** | NLP `TfidfVectorizer` and `cosine_similarity` |

---

## 📁 Project Directory Structure

```text
movie-recommendation-system/
│
├── app.py              # Main Streamlit web application
├── movies.csv          # Curated dataset of 40 movies with posters & overviews
├── requirements.txt    # Python library dependencies
├── test_app.py         # Automated unit test suite for NLP & recommendations
└── README.md           # Documentation, setup guide, and project details
```

---

## 🧠 How the Machine Learning & NLP Pipeline Works

```
[Raw Movie Data] (Title, Genre, Description, Poster)
       │
       ▼
[NLP Text Preprocessing] (Lowercasing, Regex punctuation removal, Normalization)
       │
       ▼
[Feature Engineering] (Combined weighted genres + overview text)
       │
       ▼
[TF-IDF Vectorizer] (N-grams (1,2), English stopwords removal)
       │
       ▼
[Feature Matrix] (High-dimensional sparse numerical vectors)
       │
       ▼
[Cosine Similarity] (Pairwise dot products of normalized vectors)
       │
       ▼
[Ranking & Filtering] (Top 5 highest scores, excluding the input movie)
       │
       ▼
[Streamlit UI Display] (Posters, Cards, Badges, Match Scores, Overview)
```

### 1. Feature Representation
Each movie's genres and plot description are combined into a consolidated textual representation:
```python
combined_features = clean_genre + " " + clean_genre + " " + clean_description
```
*(Genre is repeated to appropriately boost thematic importance relative to the plot summary.)*

### 2. TF-IDF (Term Frequency-Inverse Document Frequency)
Converts textual data into numerical vectors where unique keywords (e.g., *astronaut*, *superhero*, *mafia*, *dinosaurs*) receive higher weights than common English words.

### 3. Cosine Similarity Formula
Measures the cosine of the angle between two multi-dimensional vectors $A$ and $B$:
$$\text{Cosine Similarity}(A, B) = \frac{A \cdot B}{\|A\| \|B\|}$$
- Range: `0.0` (completely orthogonal/unrelated) to `1.0` (identical).
- Movies with the highest cosine scores are returned as recommendations.

---

## 🚀 Step-by-Step Guide: Running the Project in VS Code on Windows

Follow these steps to set up and run the application on Windows:

### Step 1: Open Project in VS Code
1. Launch **Visual Studio Code**.
2. Click **File** > **Open Folder...** (or press `Ctrl + K, Ctrl + O`).
3. Select the folder:
   ```text
   C:\Users\jadha\.gemini\antigravity\scratch\movie-recommendation-system
   ```

### Step 2: Open Terminal in VS Code
1. In the top menu, click **Terminal** > **New Terminal** (or press `Ctrl + \``).
2. Ensure PowerShell or Command Prompt opens in the project folder:
   ```powershell
   # Confirm you are in the project folder
   pwd
   ```

### Step 3: (Recommended) Create and Activate a Virtual Environment
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment on Windows PowerShell:
.\venv\Scripts\Activate.ps1

# (If PowerShell restricts script execution, run: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass)
```

### Step 4: Install Dependencies
Install all required libraries using `pip`:
```powershell
pip install -r requirements.txt
```

### Step 5: (Optional) Run the Test Suite
Verify that the NLP preprocessing, posters, and recommendations are functioning properly:
```powershell
python test_app.py
```
*You should see all unit tests pass with top 5 recommendations output.*

### Step 6: Launch the Streamlit Web Application
Run the Streamlit application command:
```powershell
# Standard command:
streamlit run app.py

# Or if streamlit is not in your Windows PATH:
python -m streamlit run app.py
```

### Step 7: View the App in Your Browser
Streamlit will automatically open your default browser. If it doesn't open automatically, open your browser and navigate to:
```text
http://localhost:8501
```

---

## 🎯 How to Use the Application

1. **Search or Filter**:
   - Type a keyword like *"space"*, *"crime"*, or *"batman"* in the search box to filter the movie list.
   - Click **❌ Clear Search** anytime to reset the search box.
2. **Select Movie**:
   - Select your desired movie from the dropdown list.
3. **Get Recommendations**:
   - Click the primary **🔍 Recommend Movies** button.
   - The selected movie's poster and details will appear in the blue hero card.
   - The **Top 5 similar movies** will be rendered as responsive cards with posters, match score, ratings, and genre tags.
4. **Inspect Similarity Scores**:
   - Expand the **"📊 Technical Breakdown & Cosine Similarity Scores"** section at the bottom to see exact mathematical similarity scores.
5. **Reset**:
   - Click **🔄 Reset All** to return to the catalog home view.

---

## 📊 Dataset Schema (`movies.csv`)

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `movie_id` | Integer | Unique identifier for each movie |
| `title` | String | Movie title |
| `genre` | String | Space-separated genre categories |
| `rating` | Float | IMDb viewer rating (out of 10) |
| `release_year` | Integer | Year the movie was released |
| `description` | String | Plot synopsis and narrative summary |
| `poster_url` | String | Direct URL to high-resolution movie poster |

---

## 🧪 Unit Testing

The project includes an automated test script [`test_app.py`](test_app.py) that validates:
- Dataset integrity and minimum length requirements ($\ge 30$ movies).
- `poster_url` column presence and non-empty URLs.
- Text preprocessing and TF-IDF matrix generation.
- Correct retrieval of exactly 5 recommendations.
- Non-reflexivity (the input movie is strictly excluded from its own recommendations).

To run the tests:
```powershell
python test_app.py
```

---

## 📜 License
This mini project is open-source and free to use for educational and learning purposes.
