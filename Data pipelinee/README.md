# Data Pipeline - Module 1

This project implements a raw-to-relational data pipeline that scrapes book pricing and availability data, cleans it, enriches it with currency conversion, and loads it into a normalized SQLite database. 

## Requirements
*   Python 3
*   `requests`
*   `beautifulsoup4`
*   `pandas`

## Installation and Setup
1.  Clone this repository and navigate to the `/data_pipeline` directory.
2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the pipeline:
    ```bash
    python scraper_pipeline.py
    ```

## Design Decisions and Data Cleaning

### Currency Conversion
The project uses a **fixed baseline conversion rate**:
**1 GBP = 105.50 INR**
This rate is used to compute the `price_inr` column in the database. 

### Data Cleaning
*   **Price**: We extracted only the numeric portion of the string, dropping the `£` sign, and cast it to a `float`.
*   **Ratings**: The textual star ratings (e.g., "One", "Two", "Three") were mapped to integer values `1` through `5`.
*   **Availability**: "In stock" strings were converted to boolean `True` (and loaded as `1` in SQLite), while all other states default to `False`/`0`.
*   **Messy Rows**: We chose to **drop rows** where any of the core numeric metrics (`price_gbp`, `rating`, `in_stock`) could not be parsed or were missing.
    *   *Justification*: Imputing product prices or ratings for an e-commerce intelligence pipeline could severely skew dashboard metrics. Missing data in core fields generally implies an incomplete product listing that shouldn't be benchmarked against.

## Database Schema
The database (`books.db`) is normalized into two tables sharing a primary/foreign key relationship:
1.  **`categories`**:
    *   `category_id` (INTEGER PRIMARY KEY)
    *   `category_name` (TEXT UNIQUE)
2.  **`books`**:
    *   `book_id` (INTEGER PRIMARY KEY)
    *   `title` (TEXT)
    *   `price_gbp` (REAL)
    *   `price_inr` (REAL)
    *   `rating` (INTEGER)
    *   `in_stock` (INTEGER - Boolean equivalent)
    *   `category_id` (INTEGER, Foreign Key referencing `categories`)
