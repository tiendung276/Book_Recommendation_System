# Book_Recommendation_System
Book Recommendation System
Overview
This is a simple Python project that demonstrates a book recommendation system using the K-Nearest Neighbors (KNN) algorithm. It allows users to input a book's ID and receive recommendations for similar books based on certain features.

Prerequisites
Before running the project, ensure you have the following libraries and tools installed:

Python (3.x recommended)
Pandas: pip install pandas
NumPy: pip install numpy
Scikit-Learn: pip install scikit-learn
PandasTable: pip install pandastable
Tkinter (usually included with Python installations)
Getting Started
Clone the repository or download the source code.
Ensure you have the required libraries installed by running the installation commands listed under "Prerequisites."
Place your book data in a CSV file named book_data.csv in the project directory. The CSV should contain columns for product_id, title, authors, category, n_review, avg_rating, and pages. Additional columns like original_price, current_price, quantity, manufacturer, and cover_link are also supported.
Usage
Run the project by executing the book_recommendation_system.py script.
A graphical user interface (GUI) window will open.
Enter a book's ID in the input field and click the "Ok" button.
The system will use KNN to recommend similar books based on author, quantity, category, number of reviews, average rating, and manufacturer.
Recommended books will be displayed in a table within the GUI.
Customization
You can customize the project by adjusting the KNN parameters or adding more features to the recommendation algorithm. The GUI can also be enhanced or modified to better suit your requirements.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
The project uses the Pandas, NumPy, Scikit-Learn, and PandasTable libraries, which are invaluable for data processing, machine learning, and user interface development. 
