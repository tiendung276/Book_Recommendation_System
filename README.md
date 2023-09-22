# Book Recommendation System

## Overview

This is a simple Python project that demonstrates a book recommendation system using the K-Nearest Neighbors (KNN) algorithm. It allows users to input a book's ID and receive recommendations for similar books based on certain features.

## Prerequisites

### Before running the project, ensure you have the following libraries and tools installed:

- Python (3.x recommended)
- Pandas: `pip install pandas`
- NumPy: `pip install numpy`
- Scikit-Learn: `pip install scikit-learn`
- PandasTable: `pip install pandastable`
- Tkinter (usually included with Python installations)

## Getting Started

### To get started with the project, follow these steps:

1. Clone the repository or download the source code.
2. Ensure you have the required libraries installed by running the installation commands listed under "Prerequisites."
3. Place your book data in a CSV file named `book_data.csv` in the project directory. The CSV should contain columns for `product_id`, `title`, `authors`, `category`, `n_review`, `avg_rating`, and `pages`. Additional columns like `original_price`, `current_price`, `quantity`, `manufacturer`, and `cover_link` are also supported.

## Usage

### To use the book recommendation system, follow these instructions:

1. Run the project by executing the `main.py` script.
2. A graphical user interface (GUI) window will open.
3. Enter a book's ID in the input field and click the "Ok" button.
4. The system will use KNN to recommend similar books based on author, quantity, category, number of reviews, average rating, and manufacturer.
5. Recommended books will be displayed in a table within the GUI.

## Customization

### You can customize the project by:

- Adjusting the KNN parameters.
- Adding more features to the recommendation algorithm.
- Enhancing or modifying the GUI to better suit your requirements.

## License

### This project is licensed under the MIT License.

See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The project uses the Pandas, NumPy, Scikit-Learn, and PandasTable libraries, which are invaluable for data processing, machine learning, and user interface development.
