from flask import Flask, render_template, request, jsonify, session, redirect
import pandas as pd
import pickle
import math
import sqlite3
import re
import numpy as np

app = Flask(__name__)
app.secret_key = "cinematch_secret"

# ---------------- LOAD DATA ----------------

movies = pickle.load(open('movies.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))

# ---------------- SAFETY CHECKS ----------------

if "vote_average" not in movies.columns:
    movies["vote_average"] = 0

if "poster_url" not in movies.columns:
    movies["poster_url"] = ""

if "genres" not in movies.columns:
    movies["genres"] = ""

if "release_date" not in movies.columns:
    movies["release_date"] = ""

movies["vote_average"] = pd.to_numeric(movies["vote_average"], errors="coerce").fillna(0)

MOVIES_PER_PAGE = 15

# ---------------- SQLITE DATABASE ----------------

def get_db():
    conn = sqlite3.connect("cinematch.db")
    conn.row_factory = sqlite3.Row
    return conn


def create_tables():
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS users(
        username TEXT PRIMARY KEY,
        email TEXT,
        password TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS favorites(
        username TEXT,
        movie TEXT
    )
    """)

    conn.commit()
    conn.close()


create_tables()

# ---------------- POSTER ----------------

def fetch_poster(title):
    movie = movies[movies["title"] == title]

    if not movie.empty:
        return movie.iloc[0].get("poster_url", "")

    return ""


 

# ---------------- DISPLAY CLEANERS ----------------

def add_spaces_to_name(text):
    """
    Convert CamelCase names to readable names.
    Example: JasonStatham -> Jason Statham
             RicRomanWaugh -> Ric Roman Waugh
    """
    if text is None:
        return ""
    text = str(text).strip()
    if not text:
        return ""
    text = re.sub(r'(?<!^)(?=[A-Z])', ' ', text)
    return text.strip()


def clean_list_display(value):
    """
    Convert list / numpy array / weird string list to clean comma-separated text.
    Removes brackets, quotes, extra spaces, and newlines properly.
    """

    if value is None:
        return "N/A"

    if isinstance(value, float) and pd.isna(value):
        return "N/A"

    # numpy array -> list
    if isinstance(value, np.ndarray):
        value = value.tolist()

    # normal python list
    if isinstance(value, list):
        cleaned = []
        for item in value:
            item = str(item).strip()
            item = item.replace("'", "").replace('"', "")
            item = item.replace("[", "").replace("]", "")
            item = re.sub(r"\s+", " ", item).strip()

            if item:
                cleaned.append(add_spaces_to_name(item))

        return ", ".join(cleaned) if cleaned else "N/A"

    # convert to string
    value = str(value).strip()

    if not value:
        return "N/A"

    # remove brackets + quotes
    value = value.replace("[", "").replace("]", "")
    value = value.replace("'", "").replace('"', "")

    # split by comma OR newline
    if "," in value:
        parts = [x.strip() for x in value.split(",")]
    else:
        parts = [x.strip() for x in value.splitlines()]

    cleaned = []
    for part in parts:
        part = re.sub(r"\s+", " ", part).strip()
        if part:
            cleaned.append(add_spaces_to_name(part))

    return ", ".join(cleaned) if cleaned else "N/A"


def clean_sentence(value):
    """
    Convert tokenized list into proper sentence.
    """
    if value is None:
        return "N/A"

    if isinstance(value, float) and pd.isna(value):
        return "N/A"

    if isinstance(value, np.ndarray):
        value = value.tolist()

    if isinstance(value, list):
        words = [str(v).strip() for v in value if str(v).strip()]
        return " ".join(words) if words else "N/A"

    value = str(value).strip()
    return value if value else "N/A"


# ---------------- RECOMMEND ----------------

def recommend(movie_name):
    """
    Return top 12 truly similar movies based on cosine similarity only.
    No re-sorting by rating (important).
    """

    temp_df = movies.copy()
    temp_df["title_lower"] = temp_df["title"].astype(str).str.lower().str.strip()

    movie_name = str(movie_name).lower().strip()

    if movie_name not in temp_df["title_lower"].values:
        return pd.DataFrame()

    idx = temp_df[temp_df["title_lower"] == movie_name].index[0]

    distances = similarity[idx]

    # Keep only the most similar movies
    movie_list = sorted(
    list(enumerate(distances)),
    reverse=True,
    key=lambda x: x[1]
    )[1:11]   # exactly top 10

    # IMPORTANT: Do NOT sort again by vote_average
    rec_movies = temp_df.iloc[[i[0] for i in movie_list]].copy()

    # Safe poster column
    if "poster_url" in rec_movies.columns:
        rec_movies["poster"] = rec_movies["poster_url"]
    else:
        rec_movies["poster"] = ""

    # Remove helper column
    if "title_lower" in rec_movies.columns:
        rec_movies = rec_movies.drop(columns=["title_lower"])

    return rec_movies

# ---------------- HOME ----------------

@app.route("/")
def home():

    if "user" not in session:
        return redirect("/account")

    page = request.args.get("page", 1, type=int)

    start = (page - 1) * MOVIES_PER_PAGE
    end = start + MOVIES_PER_PAGE

    page_movies = movies.iloc[start:end].copy()
    page_movies["poster"] = page_movies["poster_url"]

    total_pages = math.ceil(len(movies) / MOVIES_PER_PAGE)

    if page == 1:
       start_page = 1
       end_page = min(5, total_pages)
    else:
       start_page = page
       end_page = min(page + 3, total_pages)
    # 🔥 Trending only on first page
    trending = None
    show_trending = False

    if page == 1:
        trending = movies.copy()

        # Safe year extraction
        trending["release_year"] = pd.to_numeric(
            trending["release_date"].astype(str).str[:4],
            errors="coerce"
        )

        trending = trending[trending["release_year"] >= 2023]

        # safe numeric values
        trending["vote_average"] = pd.to_numeric(
            trending["vote_average"], errors="coerce"
        ).fillna(0)

        if "vote_count" not in trending.columns:
            trending["vote_count"] = 0

        trending["vote_count"] = pd.to_numeric(
            trending["vote_count"], errors="coerce"
        ).fillna(0)

        # Better trending logic = new + popular + rated
        trending["trending_score"] = (
            (trending["vote_average"] * 0.6) +
            (trending["vote_count"] * 0.004)
        )

        trending = trending.sort_values(
            by=["trending_score", "release_date"],
            ascending=False
        ).head(8)

        if "release_year" in trending.columns:
            trending = trending.drop(columns=["release_year"])

        if "trending_score" in trending.columns:
            trending = trending.drop(columns=["trending_score"])

        trending["poster"] = trending["poster_url"]
        show_trending = True

    return render_template(
        "index.html",
        movies=page_movies,
        trending=trending,
        show_trending=show_trending,
        similar_movies=None,
        main_movie=None,
        page=page,
        total_pages=total_pages,
        start_page=start_page,
        end_page=end_page
    )
 
# ---------------- SEARCH ----------------

@app.route("/search", methods=["POST"])
def search():

    movie_name = request.form.get("movie", "").strip()

    results = movies[movies['title'].str.contains(movie_name, case=False, na=False)].copy()

    if not results.empty:
        main_movie_title = results.iloc[0]["title"]

        main_movie = movies[movies["title"] == main_movie_title].iloc[0].copy()
        main_movie["poster"] = main_movie["poster_url"]

        recs = recommend(main_movie_title)
    else:
        main_movie = None
        recs = pd.DataFrame()

    if not results.empty:
        results["poster"] = results["poster_url"]

    return render_template(
    "index.html",
    movies=pd.DataFrame(),   # show no extra movie grid after search
    main_movie=main_movie,
    similar_movies=recs,
    trending=None,
    show_trending=False,
    page=1,
    total_pages=1,
    start_page=1,
    end_page=1
)

# ---------------- GENRE ----------------

@app.route("/genre/<genre>")
def genre(genre):

    if "tags" not in movies.columns:
        filtered = pd.DataFrame(columns=movies.columns)
    else:
        filtered = movies[movies['tags'].str.contains(genre, case=False, na=False)].copy()

    if not filtered.empty:
        filtered["poster"] = filtered["poster_url"]

    return render_template(
        "index.html",
        movies=filtered,
        trending=None,
        show_trending=False,
        main_movie=None,
        similar_movies=None,
        page=1,
        total_pages=1,
        start_page=1,
        end_page=1
    )

# ---------------- LANGUAGE FILTER ----------------

@app.route("/language/<lang>")
def by_language(lang):

    if "user" not in session:
        return redirect("/account")

    page = request.args.get("page", 1, type=int)

    if page < 1:
        page = 1

    # Accept URL values from navbar
    lang = lang.strip()

    # Allowed language mapping
    language_map = {
        "English": ["English", "en"],
        "Hindi": ["Hindi", "hi"]
    }

    # Invalid language? go home
    if lang not in language_map:
        return redirect("/")

    # Make copy
    filtered = movies.copy()

    # IMPORTANT: your dataset uses 'languages' column
    if "languages" not in filtered.columns:
        return redirect("/")

    # Get possible match values
    possible_values = language_map[lang]

    # Filter safely (works for English/Hindi or en/hi inside strings/lists)
    mask = False
    for value in possible_values:
        mask = mask | filtered["languages"].astype(str).str.contains(value, case=False, na=False)

    filtered = filtered[mask].copy()

    # Add poster safely
    if "poster_url" in filtered.columns:
        filtered["poster"] = filtered["poster_url"]
    else:
        filtered["poster"] = ""

    # Pagination
    total_pages = max(1, math.ceil(len(filtered) / MOVIES_PER_PAGE))

    if page > total_pages:
        page = total_pages

    start = (page - 1) * MOVIES_PER_PAGE
    end = start + MOVIES_PER_PAGE

    page_movies = filtered.iloc[start:end].copy()

    # Pagination style (same as homepage)
    if page == 1:
        start_page = 1
        end_page = min(5, total_pages)
    else:
        start_page = page
        end_page = min(page + 3, total_pages)

    return render_template(
        "index.html",
        movies=page_movies,
        trending=None,
        show_trending=False,
        similar_movies=None,
        main_movie=None,
        page=page,
        total_pages=total_pages,
        start_page=start_page,
        end_page=end_page,
        current_filter=f"Language: {lang}"
    )

# ---------------- YEAR ----------------

@app.route("/year/<year>")
def year(year):

    filtered = movies[movies['release_date'].astype(str).str.startswith(year)].copy()

    if not filtered.empty:
        filtered["poster"] = filtered["poster_url"]

    return render_template(
        "index.html",
        movies=filtered,
        trending=None,
        show_trending=False,
        main_movie=None,
        similar_movies=None,
        page=1,
        total_pages=1,
        start_page=1,
        end_page=1
    )

# ---------------- ANIMATION ----------------

@app.route("/genre/Animation")
def animation():

    if "tags" not in movies.columns:
        filtered = pd.DataFrame(columns=movies.columns)
    else:
        filtered = movies[movies['tags'].str.contains("animation", case=False, na=False)].copy()

    if not filtered.empty:
        filtered["poster"] = filtered["poster_url"]

    return render_template(
        "index.html",
        movies=filtered,
        trending=None,
        show_trending=False,
        main_movie=None,
        similar_movies=None,
        page=1,
        total_pages=1,
        start_page=1,
        end_page=1
    )
  # ---------------- MOVIE DETAILS ----------------

@app.route("/movie/<title>")
def movie_detail(title):

    movie_rows = movies[movies["title"] == title]

    if movie_rows.empty:
        return "Movie not found"

    movie = movie_rows.iloc[0].copy()

    # safe poster
    movie["poster"] = movie["poster_url"] if "poster_url" in movie.index else ""

    # clean fields for display
    if "tagline" in movie.index:
        movie["tagline"] = clean_sentence(movie["tagline"])
    else:
        movie["tagline"] = "N/A"

    if "overview" in movie.index:
        movie["overview"] = clean_sentence(movie["overview"])
    else:
        movie["overview"] = "N/A"

    if "genres" in movie.index:
        movie["genres"] = clean_list_display(movie["genres"])
    else:
        movie["genres"] = "N/A"

    if "spoken_languages" in movie.index:
        movie["spoken_languages"] = clean_list_display(movie["spoken_languages"])
    else:
        movie["spoken_languages"] = "N/A"

    # cast and director
    movie["genres"] = clean_list_display(movie["genres"]) if "genres" in movie.index else "N/A"
    movie["spoken_languages"] = clean_list_display(movie["spoken_languages"]) if "spoken_languages" in movie.index else "N/A"

    cast = clean_list_display(movie["cast"]) if "cast" in movie.index else "N/A"
    director = clean_list_display(movie["director"]) if "director" in movie.index else "N/A"

    # recommendations
    recs = recommend(title)

    return render_template(
        "details.html",
        movie=movie,
        cast=cast,
        director=director,
        recs=recs
    )

# ---------------- FAVORITES ----------------

@app.route("/favorite", methods=["POST"])
def favorite():

    if "user" not in session:
        return jsonify({"status": "login_required"})

    title = request.json.get("title")
    username = session["user"]

    conn = get_db()

    conn.execute(
        "INSERT INTO favorites(username,movie) VALUES (?,?)",
        (username, title)
    )

    conn.commit()
    conn.close()

    return jsonify({"status": "ok"})

# ---------------- FAVORITE PAGE ----------------

@app.route("/favorites")
def show_favorites():

    if "user" not in session:
        return redirect("/account")

    username = session["user"]

    conn = get_db()

    fav_list = conn.execute(
        "SELECT movie FROM favorites WHERE username=?",
        (username,)
    ).fetchall()

    conn.close()

    titles = [x["movie"] for x in fav_list]

    fav_movies = movies[movies["title"].isin(titles)].copy()

    if not fav_movies.empty:
        fav_movies["poster"] = fav_movies["poster_url"]

    return render_template("favorites.html", movies=fav_movies)

# ---------------- ACCOUNT ----------------

@app.route("/account")
def account():

    if "user" in session:
        return redirect("/")

    return render_template("account.html")

@app.route("/signup", methods=["POST"])
def signup():

    username = request.form["username"]
    email = request.form["email"]
    password = request.form["password"]

    conn = get_db()

    conn.execute(
        "INSERT OR IGNORE INTO users VALUES (?,?,?)",
        (username, email, password)
    )

    conn.commit()
    conn.close()

    session["user"] = username

    return redirect("/")

@app.route("/login", methods=["POST"])
def login():

    username = request.form["username"]
    password = request.form["password"]

    conn = get_db()

    user = conn.execute(
        "SELECT * FROM users WHERE username=? AND password=?",
        (username, password)
    ).fetchone()

    conn.close()

    if user:
        session["user"] = username
        return redirect("/")

    return redirect("/account")

@app.route("/logout")
def logout():

    session.pop("user", None)
    return redirect("/account")
 

 # ---------------- PROFILE ----------------

@app.route("/profile")
def profile():

    if "user" not in session:
        return redirect("/account")

    username = session["user"]

    conn = get_db()

    user = conn.execute(
        "SELECT * FROM users WHERE username=?",
        (username,)
    ).fetchone()

    fav_count = conn.execute(
        "SELECT COUNT(*) as total FROM favorites WHERE username=?",
        (username,)
    ).fetchone()["total"]

    conn.close()

    return render_template("profile.html", user=user, fav_count=fav_count) 

# ---------------- ABOUT PAGE ----------------

@app.route("/about")
def about():
    if "user" not in session:
        return redirect("/account")
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)