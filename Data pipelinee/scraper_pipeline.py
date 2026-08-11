import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3

def scrape_books():
    print("Scraping books...")
    books_data = []
    page = 1
    
    # Scrape until we have at least 65 books to ensure we easily hit the >60 requirement.
    while len(books_data) < 65:
        url = f"https://books.toscrape.com/catalogue/page-{page}.html"
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to fetch page {page}")
            break
            
        soup = BeautifulSoup(response.content, 'html.parser')
        books = soup.find_all('article', class_='product_pod')
        
        for book in books:
            title = book.h3.a['title']
            price_text = book.find('p', class_='price_color').text
            
            star_classes = book.find('p', class_='star-rating')['class']
            star_rating = [c for c in star_classes if c != 'star-rating'][0]
            
            # Follow product link to get availability and category
            product_url = "https://books.toscrape.com/catalogue/" + book.h3.a['href']
            prod_response = requests.get(product_url)
            prod_soup = BeautifulSoup(prod_response.content, 'html.parser')
            
            availability = prod_soup.find('p', class_='availability').text.strip()
            
            breadcrumb = prod_soup.find('ul', class_='breadcrumb')
            category = breadcrumb.find_all('li')[2].text.strip()
            
            books_data.append({
                'title': title,
                'price': price_text,
                'star_rating': star_rating,
                'availability': availability,
                'category': category
            })
            
            if len(books_data) >= 65:
                break
                
        page += 1
        
    categories_count = len(set([b['category'] for b in books_data]))
    print(f"Scraped {len(books_data)} books across {categories_count} categories.")
    
    # Explicit requirement checks
    if len(books_data) < 60:
        raise ValueError("Requirement failed: fewer than 60 books scraped.")
    if categories_count < 3:
        raise ValueError("Requirement failed: fewer than 3 categories scraped.")
        
    return pd.DataFrame(books_data)

def clean_and_convert(df):
    print("Cleaning and converting data...")
    # Strip currency symbol and convert to float
    df['price_gbp'] = df['price'].str.extract(r'([\d.]+)').astype(float)
    
    # Convert star rating to integer robustly
    rating_map = {'One': 1, 'Two': 2, 'Three': 3, 'Four': 4, 'Five': 5}
    df['rating'] = pd.to_numeric(df['star_rating'].map(rating_map), errors='coerce')
    
    # Parse availability to boolean
    df['in_stock'] = df['availability'].str.contains('In stock', case=False, na=False).astype(bool)
    
    # Handle messy rows
    df.dropna(subset=['title', 'price_gbp', 'rating', 'category'], inplace=True)
    df['rating'] = df['rating'].astype(int)
    
    # Convert GBP to INR using fixed baseline
    # 1 GBP = 105.50 INR
    df['price_inr'] = (df['price_gbp'] * 105.50).round(2)
    
    return df

def load_to_db(df, db_path='books.db'):
    print(f"Loading data into {db_path}...")
    conn = sqlite3.connect(db_path)
    # Enable foreign key enforcement
    conn.execute("PRAGMA foreign_keys = ON")
    cursor = conn.cursor()
    
    # Design normalized SQLite schema
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS categories(
            category_id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_name TEXT UNIQUE
        )
    ''')
    
    # Drop existing books table to ensure fresh insertion
    cursor.execute('DROP TABLE IF EXISTS books')
    cursor.execute('''
        CREATE TABLE books(
            book_id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            price_gbp REAL,
            price_inr REAL,
            rating INTEGER,
            in_stock INTEGER,
            category_id INTEGER,
            FOREIGN KEY(category_id) REFERENCES categories(category_id)
        )
    ''')
    
    # Insert categories
    unique_categories = df[['category']].drop_duplicates()
    for _, row in unique_categories.iterrows():
        cursor.execute('INSERT OR IGNORE INTO categories (category_name) VALUES (?)', (row['category'],))
        
    # Get category mapping
    category_map = pd.read_sql('SELECT category_id, category_name FROM categories', conn)
    
    # Map category_id back to df
    df = df.merge(category_map, left_on='category', right_on='category_name', how='left')
    
    # Insert books preserving the schema
    books_to_insert = df[['title', 'price_gbp', 'price_inr', 'rating', 'in_stock', 'category_id']]
    books_to_insert.to_sql('books', conn, if_exists='append', index=False)
    
    conn.commit()
    conn.close()

def run_queries_and_verify(db_path='books.db'):
    print("Running SQL queries...")
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    
    q1 = """
        SELECT title, price_gbp, rating
        FROM books
        WHERE rating >= 4
        ORDER BY price_gbp DESC
        LIMIT 5;
    """
    
    q2 = """
        SELECT DISTINCT rating
        FROM books
        ORDER BY rating DESC;
    """
    
    q3 = """
        SELECT title, rating
        FROM books
        WHERE rating IN (1, 5)
        LIMIT 5;
    """
    
    q4 = """
        SELECT title, price_inr
        FROM books
        WHERE price_inr BETWEEN 2000 AND 3000
        LIMIT 5;
    """
    
    q5 = """
        SELECT c.category_name, b.title, b.rating
        FROM books b
        JOIN categories c ON b.category_id = c.category_id
        ORDER BY b.rating DESC, c.category_name ASC
        LIMIT 10;
    """
    
    results = {
        "Query 1 - SELECT WHERE ORDER BY LIMIT": pd.read_sql(q1, conn),
        "Query 2 - DISTINCT": pd.read_sql(q2, conn),
        "Query 3 - IN": pd.read_sql(q3, conn),
        "Query 4 - BETWEEN": pd.read_sql(q4, conn),
        "Query 5 - JOIN": pd.read_sql(q5, conn)
    }

    queries = {
        "Query 1 - SELECT WHERE ORDER BY LIMIT": q1,
        "Query 2 - DISTINCT": q2,
        "Query 3 - IN": q3,
        "Query 4 - BETWEEN": q4,
        "Query 5 - JOIN": q5
    }

    # Save queries and outputs
    with open("sql_query_results.txt", "w", encoding="utf-8") as f:
        for name in queries:
            f.write("=" * 80 + "\n")
            f.write(name + "\n")
            f.write("=" * 80 + "\n")
            f.write(queries[name] + "\n")
            f.write("OUTPUT:\n")
            f.write(results[name].to_string(index=False))
            f.write("\n\n")

    print("SQL queries and outputs saved to sql_query_results.txt")
    
    # Pandas merge reproduction
    print("\n--- Pandas Merge Reproduction ---")
    books_df = pd.read_sql("SELECT * FROM books", conn)
    categories_df = pd.read_sql("SELECT * FROM categories", conn)
    
    merged_df = pd.merge(books_df, categories_df, on='category_id', how='inner')
    merged_df = merged_df[['category_name', 'title', 'rating']]
    merged_df = merged_df.sort_values(by=['rating', 'category_name'], ascending=[False, True]).head(10).reset_index(drop=True)
    
    sql_join_df = results["Query 5 - JOIN"]
    
    print("\nDo the outputs match?")
    print(sql_join_df.equals(merged_df))
    
    conn.close()

if __name__ == "__main__":
    raw_df = scrape_books()
    clean_df = clean_and_convert(raw_df)
    load_to_db(clean_df)
    run_queries_and_verify()
