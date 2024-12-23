import os
from typing import List, Tuple, Union, Dict
#
from pymilvus import settings as milvus_settings
from pymilvus import connections, db, MilvusClient, DataType, FieldSchema, CollectionSchema, DataType, utility


class DatabaseClient:
    def __init__(
        self,
        host: str,
        port: str,
        scheme: str = "http",
        db_name: str = "songs",
    ):
        self.host = host
        self.port = port
        self.scheme = scheme
        self.db_name = db_name
        self.db_alias = milvus_settings.Config.MILVUS_CONN_ALIAS

    @property
    def should_setup_from_scratch(self):
        return len(self.client.list_collections()) == 0

    @property
    def song_collections(self) -> Tuple:
        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="id", index_type="STL_SORT")
        index_params.add_index(
            field_name="embedding",
            index_type="FLAT",
            metric_type="COSINE",
            params={"nlist": 128}
        )
        schema = CollectionSchema(
            fields=[
                FieldSchema(
                    name='id',
                    is_primary=True,
                    dtype=DataType.INT64,
                    description="The id of the song."
                ),
                FieldSchema(
                    name='embedding',
                    dtype=DataType.FLOAT_VECTOR,
                    description="The embedding of audio.",
                    dim=2048,
                ),
                FieldSchema(
                    name='song_title',
                    dtype=DataType.VARCHAR,
                    max_length=256,
                    description="The original audio title.",
                ),
                FieldSchema(
                    name='start_position_sec',
                    dtype=DataType.INT32,
                    description="The start position of the chunk in sec.",
                ),
                FieldSchema(
                    name='end_position_sec',
                    dtype=DataType.INT32,
                    description="The end position of the chunk in sec.",
                ),
                FieldSchema(
                    name='song_id',
                    dtype=DataType.INT64,
                    nullable=True,
                    description="The ID of the original audio.",
                ),
                FieldSchema(
                    name='is_chunk',
                    dtype=DataType.BOOL,
                    description="The ID of the original audio.",
                ),
                FieldSchema(
                    name='audio_url',
                    dtype=DataType.VARCHAR,
                    max_length=256,
                    description="The ID of the original audio.",
                ),
            ],

            auto_id=True,
            enable_dynamic_field=True,
            description=f"All the registered songs.",
        )
        return "songs", schema, index_params

    @property
    def collections(self) -> List[Tuple]:
        return [
            self.song_collections,
        ]

    def connect(self):
        """ Connects to the Milvus database server """
        connections.connect(
            host=self.host,
            port=self.port,
            alias=self.db_alias
        )

        if self.db_name not in db.list_database():
            db.create_database(self.db_name)
        db.using_database(self.db_name)
        self.client = MilvusClient(
            uri=f"{self.scheme}://{self.host}:{self.port}",
            db_name=self.db_name,
        )

    def disconnect(self):
        """ Disconnects from the Milvus database server """
        self.client.close()
        connections.disconnect(milvus_settings.Config.MILVUS_CONN_ALIAS)

    def load_collections(self):
        print("Loading collections")
        for name, _, _ in self.collections:
            print(f"-- {name}")
            self.client.load_collection(name)

    def release_collections(self):
        print("Releasing collections")
        for name, _, _, _ in self.collections:
            print(f"-- {name}")
            self.client.release_collection(name)

    def setup_from_scratch(self):
        """ Create the collections """
        print("Create the different collections.")
        for name, schema, index_params in self.collections:
            print(f"-- {name}")
            self.client.create_collection(
                name,
                schema=schema,
                index_params=index_params,
            )

    def drop_db(self):
        print("dropping all collections")
        for name, _, _ in self.collections:
            print(f"-- {name}")
            self.client.drop_collection(name)

    def insert(
        self,
        collection_name: str,
        data: Union[Dict, List[Dict]],
    ) -> Dict:
        return self.client.insert(
            collection_name=collection_name,
            data=data,
        )
