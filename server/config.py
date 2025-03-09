# config.py
import os

class Config:
    # Set your MySQL connection URI
    SQLALCHEMY_DATABASE_URI = 'mysql+mysqlconnector://superuser:superuser@localhost/dynamic_tables'
    SQLALCHEMY_TRACK_MODIFICATIONS = False