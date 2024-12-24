import os
from database_client import DatabaseClient


if __name__ == "__main__":
    client = DatabaseClient(
        host="134.209.97.57",
        port="19530",

    )
    client.connect()

    client.drop_db()

    print("Drop collections success.")
    client.disconnect()
