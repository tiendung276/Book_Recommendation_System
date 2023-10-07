import pandas as pd
import sklearn
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from collections import Counter

# customer data
customer_data = pd.read_csv("data//comments.csv", sep=",").iloc[:, :-1]
customer_data = customer_data.drop_duplicates()
le = LabelEncoder()

customer_id = customer_data["customer_id"]
book_id = customer_data["product_id"]
cmt_id = customer_data["comment_id"]
rating = customer_data["rating"]
thank_count = customer_data["thank_count"]

# BOOK DATA
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

title = book_data["title"]

merged_data = pd.merge(customer_data, book_data, on='product_id').drop_duplicates()

customer_to_recommend = 27

high_rated_books = merged_data[(merged_data['customer_id'] == customer_to_recommend) & (merged_data['rating'] >= 4)][
    'product_id']

book_features = list(
    zip(authors, category, manufacturer, original_price, current_price, quantity, n_review, avg_rating, pages))
knn = NearestNeighbors(n_neighbors=7, algorithm='ball_tree').fit(book_features)
distances, indices = knn.kneighbors(book_features)

recommended_books = []
for book_id in high_rated_books:
    book_idx = np.where(product_id == book_id)[0][0]
    for i in indices[book_idx][1:]:
        recommended_books.append(product_id[i])
print(f"Những cuốn sách mà người dùng {customer_to_recommend} đã đọc :")

for i in range(merged_data.shape[0]):
    if merged_data.loc[i, "customer_id"] == customer_to_recommend:
        print(merged_data.loc[i, "title_y"], "|Thể Loại:", merged_data.loc[i, "category"], "|Tác Giả:",
              merged_data.loc[i, "authors"])

print("\nSách được đề xuất cho khách hàng", customer_to_recommend, "là:")
for book_id in recommended_books:
    print(book_data["title"][np.where(product_id == book_id)[0][0]], "|Thể Loại:",
          book_data["category"][np.where(product_id == book_id)[0][0]], "|Tác Giả:",
          book_data["authors"][np.where(product_id == book_id)[0][0]])

"""    print(title[np.where(product_id == book_id)[0][0]])
"""
