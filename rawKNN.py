import pandas as pd
import sklearn
from sklearn.neighbors import NearestNeighbors

book_data = pd.read_csv("data\\book_data.csv", sep=",")

book_data = book_data.drop_duplicates()

le = sklearn.preprocessing.LabelEncoder()

pages = pd.to_numeric(book_data['pages'], errors='coerce').fillna(180)
book_data["avg_rating"] = book_data["avg_rating"].fillna(0)
book_data["quantity"] = book_data["quantity"].fillna(0)
book_data["authors"] = book_data["authors"].fillna('Unknown')
book_data["manufacturer"] = book_data["manufacturer"].fillna("Unknown")

authors = le.fit_transform(list(book_data['authors']))
category = le.fit_transform(list(book_data["category"]))
manufacturer = le.fit_transform(list(book_data["manufacturer"]))

original_price = book_data["original_price"]
current_price = book_data["current_price"]
quantity = book_data["quantity"]
n_review = book_data["n_review"]
avg_rating = book_data["avg_rating"]

X = list(zip(authors, quantity, category, n_review, avg_rating, manufacturer))

book_to_recommend = int(input("ID of book to recommend: "))
book_to_recommend_index = book_data[book_data["product_id"] == book_to_recommend].index[0]

knn = NearestNeighbors(n_neighbors=7, metric='euclidean')
knn.fit(X)

distances, indices = knn.kneighbors([X[book_to_recommend_index]])

recommended_books_indices = indices[0][1:]
recommended_books = book_data.iloc[recommended_books_indices]

features = ["authors", "quantity", "category", "n_review", "avg_rating", "manufacturer"]
print(
    f"Book to recommend: {book_data.loc[book_to_recommend_index, 'title']} by {book_data.loc[book_to_recommend_index, 'authors']}")
print("\nRecommended books:")
print(recommended_books[features].to_string(index=False))
