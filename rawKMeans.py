import pandas as pd
import sklearn
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans

book_data = pd.read_csv("data\\book_data.csv", sep=",")

"""Data Preprocessing"""
book_data = book_data.drop_duplicates()

le = sklearn.preprocessing.LabelEncoder()

pages = (pd.to_numeric(book_data['pages'], errors='coerce')).fillna(180)
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

book_to_recommend = 73787185
"""int(input("ID of book to recommend: "))"""
book_to_recommend_index = book_data[book_data["product_id"] == book_to_recommend].index[0]

knn = NearestNeighbors(n_neighbors=9, metric='euclidean')
knn.fit(X)
distances, indices = knn.kneighbors([X[book_to_recommend_index]])
recommended_books_indices_knn = indices[0][1:]
recommended_books_knn = book_data.iloc[recommended_books_indices_knn]

kmeans = KMeans(n_clusters=7, n_init=10, random_state=0).fit(X)
labels = kmeans.labels_
cluster_to_recommend = labels[book_to_recommend_index]
recommended_books_indices_kmeans = [i for i, label in enumerate(labels) if
                                    label == cluster_to_recommend and i != book_to_recommend_index]
recommended_books_kmeans = book_data.iloc[recommended_books_indices_kmeans]

features = ["authors", "title", "quantity", "category", "n_review", "avg_rating", "manufacturer"]
print(
    f"Book to recommend: {book_data.loc[book_to_recommend_index, 'title']} by {book_data.loc[book_to_recommend_index, 'authors']}")
print("\nRecommended books by KNN:")
print(recommended_books_knn[features].to_string(index=False))
print("\nRecommended books by KMeans:")
print(recommended_books_kmeans[features][:9].to_string(index=False))

recommended_books_indices_knn_set = set(recommended_books_indices_knn)
recommended_books_indices_kmeans_set = set(recommended_books_indices_kmeans[:9])
common_indices = recommended_books_indices_knn_set & recommended_books_indices_kmeans_set

print(
    f"The number of recommended books suggested by KNN falls within the same cluster as the book to be recommended according to K-Means: {len(common_indices)} / 9")
