import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.cluster import KMeans

# customer data
customer_data = pd.read_csv("comments.csv", sep=",")
customer_data.drop_duplicates(inplace=True)
le = LabelEncoder()

customer_id = customer_data["customer_id"]
book_id = customer_data["product_id"]
cmt_id = customer_data["comment_id"]
rating = customer_data["rating"]
thank_count = customer_data["thank_count"]

# book data
book_data = pd.read_csv("book_data.csv", sep=",").iloc[:, :-1]
book_data.drop_duplicates(inplace=True)

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

book_data["pages"] = book_data["pages"].fillna(180)
pages = (pd.to_numeric(book_data['pages'], errors='coerce'))

product_id = book_data["product_id"]

title = book_data["title"]

merged_data = pd.merge(customer_data, book_data, on='product_id').drop_duplicates()

customer_to_recommend = 20363628

high_rated_books = merged_data[(merged_data['customer_id'] == customer_to_recommend) & (merged_data['rating'] >= 4)][
    'product_id']

book_features = list(zip(authors, category, manufacturer, quantity, n_review, avg_rating, pages))

kmeans = KMeans(n_clusters=7, n_init=10, random_state=0).fit(book_features)
labels = kmeans.labels_

recommended_books_kmeans_indices = []
for book_id in high_rated_books:
    book_data.reset_index(drop=True, inplace=True)
    book_idx = np.where(product_id == book_id)[0][0]
    cluster_to_recommend = labels[book_idx]
    recommended_books_kmeans_indices.extend(
        [i for i, label in enumerate(labels) if label == cluster_to_recommend and i != book_idx])

recommended_books_kmeans = book_data.iloc[recommended_books_kmeans_indices]

print(f"Books that user {customer_to_recommend} has read:")
for i in range(merged_data.shape[0]):
    if merged_data.loc[i, "customer_id"] == customer_to_recommend:
        print(merged_data.loc[i, "title_y"], " |Category:", merged_data.loc[i, "category"], " |Author:",
              merged_data.loc[i, "authors"], " |Rating", merged_data.loc[i, "rating"])


print("\nBooks recommended for customer", customer_to_recommend, "are:")
for i in recommended_books_kmeans_indices:
    print(book_data.loc[i, "title"], "|Category:",
          book_data.loc[i, "category"], "|Author:",
          book_data.loc[i, "authors"])
