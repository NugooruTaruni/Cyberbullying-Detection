from pymongo import MongoClient
import ssl
from dotenv import load_dotenv
load_dotenv()

connection_string="mongodb+srv://TaruniNugooru:{os.getenv('MONGO_PASSWORD')}@cluster0.lu57udn.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client=MongoClient(connection_string,ssl.CERT_NONE)
dataSeesaws=client.CBDetection
collection=dataSeesaws.Users
Comments = dataSeesaws.Comments