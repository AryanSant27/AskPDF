
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

uri = "mongodb+srv://santaryan27:%40Aryan27@askpdf.zntzict.mongodb.net/?retryWrites=true&w=majority&appName=AskPDF" # This file was for a one-time test. Use app.py to run the application.

# Create a new client and connect to the server
client = MongoClient(uri)

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)