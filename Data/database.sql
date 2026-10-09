CREATE DATABASE kinbech_db;
USE kinbech_db;

-- Users table
CREATE TABLE users (
    user_id INT PRIMARY KEY AUTO_INCREMENT,
    username VARCHAR(50) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,
    email VARCHAR(100) UNIQUE,
    role ENUM('user', 'admin') DEFAULT 'user',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Categories table
CREATE TABLE categories (
    category_id INT PRIMARY KEY AUTO_INCREMENT,
    category_name VARCHAR(100) NOT NULL,
    description TEXT
);

-- Product_Listing table
CREATE TABLE product_listing (
    listing_id INT PRIMARY KEY AUTO_INCREMENT,
    title VARCHAR(150) NOT NULL,
    category_id INT NOT NULL,
    original_price DECIMAL(10,2) NOT NULL,
    asking_price DECIMAL(10,2) NOT NULL,
    used_months INT NOT NULL,
    condition_score INT NOT NULL,
    brand VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (category_id) REFERENCES categories(category_id)
);

-- Price_Prediction_Log table
CREATE TABLE price_prediction_log (
    log_id INT PRIMARY KEY AUTO_INCREMENT,
    listing_id INT,
    predicted_price DECIMAL(10,2) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (listing_id) REFERENCES product_listing(listing_id)
);
