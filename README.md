# Overview
This Python project showcases a book recommendation system using the K-Nearest Neighbors (KNN) and KMeans algorithms. Users can input a book's ID to receive recommendations for similar books based on specific features.

# Prerequisites
Before running the project, make sure you have the following libraries and tools installed:
- Python (3.x recommended)
- Pandas: pip install pandas
- Scikit-Learn: pip install scikit-learn
- PandasTable: pip install pandastable
- Tkinter (usually included with Python installations)

# Getting Started
To begin with the project, follow these steps:
1. Clone the repository or download the source code.
2. Verify that you have the necessary libraries installed by running the commands mentioned under "Prerequisites."
3. Organize your book data in a CSV file named book_data.csv within the project directory. The CSV should include columns for product_id, title, authors, category, n_review, avg_rating, and pages. You can also include additional columns like original_price, current_price, quantity, manufacturer, and cover_link.

# Usage
Using the book recommendation system:
1. Run the project by executing the main(GUI).py script.
2. A graphical user interface (GUI) window will appear.
3. Input a book's ID into the provided field and click the "Ok" button.
4. The system employs KNN to suggest similar books based on attributes such as author, quantity, category, number of reviews, average rating, and manufacturer.
5. The recommended books will be displayed in a table within the GUI.

Note: You have the option to run the system using rawKMeans.py or rawKNN.py without the GUI. Additionally, you can use BRS_with_user_data.py to recommend books based on user data.

# Customization
You can tailor the project to your needs by:
- Tweaking the KNN parameters.
- Introducing additional features to enhance the recommendation algorithm.
- Modifying or enhancing the GUI to better match your specific requirements.

# License
This project is licensed under the MIT License.
Refer to the LICENSE file for more details.

# Acknowledgments
This project relies on Pandas, NumPy, Scikit-Learn, and PandasTable libraries, which are indispensable for data processing, machine learning, and user interface development.
