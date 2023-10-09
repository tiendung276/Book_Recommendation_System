import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.neighbors import NearestNeighbors

# customer data
customer_data = pd.read_csv("data//comments.csv", sep=",").iloc[:, :-1]
customer_data = customer_data.drop_duplicates()
le = LabelEncoder()

comment_title = customer_data["title"]
customer_id = customer_data["customer_id"]
book_id = customer_data["product_id"]
cmt_id = customer_data["comment_id"]
rating = customer_data["rating"]
thank_count = customer_data["thank_count"]

# book data
book_data = pd.read_csv("data//book_data.csv", sep=",")
book_data = book_data.drop_duplicates()

book_data["authors"] = book_data["authors"].fillna('Unknown')
authors = le.fit_transform(list(book_data['authors']))

index = book_data["category"].str.lower().str.contains('tập')
book_data.loc[index, 'category'] = 'Truyện Tranh'
category = le.fit_transform(list(book_data["category"]))

book_data["manufacturer"] = book_data["manufacturer"].fillna("Unknown")
manufacturer = le.fit_transform(list(book_data["manufacturer"]))

original_price = book_data["original_price"]

current_price = book_data["current_price"]

book_data["quantity"] = book_data["quantity"].fillna(0)
quantity = book_data["quantity"]

n_review = book_data["n_review"]

book_data["avg_rating"] = book_data["avg_rating"].fillna(0)
avg_rating = book_data["avg_rating"]

pages = (pd.to_numeric(book_data['pages'], errors='coerce')).fillna(180)

product_id = book_data["product_id"]

book_title = book_data["title"]

merged_data = pd.merge(customer_data, book_data, on='product_id').drop_duplicates()

customer_to_recommend = merged_data['customer_id'].sample(n=1).iloc[0]

print(f"Books that user {customer_to_recommend} has read:")
for i in range(merged_data.shape[0]):
    if merged_data.loc[i, "customer_id"] == customer_to_recommend:
        print(merged_data.loc[i, "title_y"], " |Category:", merged_data.loc[i, "category"], " |Author:",
              merged_data.loc[i, "authors"], " |Rating", merged_data.loc[i, "rating"])

high_rated_books = merged_data[(merged_data['customer_id'] == customer_to_recommend) & (merged_data['rating'] >= 4)][
    'product_id']

book_features = list(zip(authors, category, manufacturer, quantity, n_review, avg_rating, pages))
knn = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(book_features)
distances, indices = knn.kneighbors(book_features)

recommended_books = []
for book_id in high_rated_books:
    book_idx = np.where(product_id == book_id)[0][0]
    for i in indices[book_idx][1:]:
        recommended_books.append(product_id.iloc[i])

print("\nBooks recommended for customer", customer_to_recommend, "are:")
for book_id in recommended_books:
    book_data = book_data.reset_index(drop=True)
    print(book_data["title"][np.where(product_id == book_id)[0][0]], "|Category:",
          book_data["category"][np.where(product_id == book_id)[0][0]], "|Author:",
          book_data["authors"][np.where(product_id == book_id)[0][0]])
