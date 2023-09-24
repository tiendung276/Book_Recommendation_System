# Import các thư viện cần thiết
import pandas as pd
import numpy as np
import sklearn
from sklearn.neighbors import NearestNeighbors
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import tkinter as tk
from pandastable import Table, TableModel

book_data = pd.read_csv("book_data.csv", sep=",")

"""Data Preprocessing"""
book_data = book_data.drop_duplicates()
le = sklearn.preprocessing.LabelEncoder()
book_data["authors"] = book_data["authors"].fillna('Unknown')
authors = le.fit_transform(list(book_data['authors']))
category = le.fit_transform(list(book_data["category"]))
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

X = list(zip(authors, quantity, category, n_review, avg_rating, manufacturer))

print("ID of book to recommend: ")
book_to_recommend = int(input())
book_to_recommend_index = book_data[book_data["product_id"] == int(book_to_recommend)].index[0]
knn = NearestNeighbors(n_neighbors=7, metric='euclidean')
knn.fit(X)
book_features = X[book_to_recommend_index]
distances, indices = knn.kneighbors([book_features])

recommended_books_indices = indices[0][1:]
recommended_books = book_data.iloc[recommended_books_indices]

print(
    recommended_books[["title", "authors", "quantity", "category", "n_review", "avg_rating", "manufacturer"]].to_string(
        index=False))
