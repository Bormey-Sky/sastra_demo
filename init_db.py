from database_client import DatabaseClient

client = DatabaseClient(
    host="127.0.0.1",
    port="19530",
    
)

client.connect()

if client.should_setup_from_scratch: 
    client.setup_from_scratch()
    print("setting up from scratch")
client.disconnect()