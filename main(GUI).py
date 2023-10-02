import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
import tkinter as tk
from pandastable import Table, TableModel

book_data = pd.read_csv("data\\book_data.csv", sep=",")
book_data = book_data.drop_duplicates()
le = LabelEncoder()
book_data["authors"] = book_data["authors"].fillna('Unknown')
book_data["quantity"] = book_data["quantity"].fillna(0)
book_data["avg_rating"] = book_data["avg_rating"].fillna(0)
book_data['pages'] = (pd.to_numeric(book_data['pages'], errors='coerce')).fillna(180)

for col in ["authors", "category", "manufacturer"]:
    book_data[col] = le.fit_transform(list(book_data[col]))

root = tk.Tk()
root.geometry("1280x600")

frame = tk.Frame(root)
frame.pack(fill='both', expand=True)


def recommend_books():
    user_input = entry.get()

    X = book_data[["authors", "quantity", "category", "n_review", "avg_rating", "manufacturer"]].values

    if user_input:
        book_to_recommend_index = book_data[book_data["product_id"] == int(user_input)].index[0]
        knn = NearestNeighbors(n_neighbors=7, metric='euclidean')
        knn.fit(X)
        book_features = X[book_to_recommend_index]
        distances, indices = knn.kneighbors([book_features])

        recommended_books_indices = indices[0][1:]
        recommended_books = book_data.iloc[recommended_books_indices]

        pt.updateModel(TableModel(recommended_books))
    else:
        pt.updateModel(TableModel(book_data))

    pt.redraw()


label = tk.Label(root, text="Nhập ID:")
label.pack(side="left")

entry = tk.Entry(root)
entry.pack(side="left")

button = tk.Button(root, text="Ok", command=recommend_books)
button.pack(side="left")

pt = Table(frame, dataframe=book_data[["product_id", "title", "authors", "category", "n_review", "avg_rating"]])
pt.show()

root.mainloop()
