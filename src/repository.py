import os
import pandas as pd

from sqlalchemy import create_engine, Engine, URL
from dotenv import load_dotenv
from src.logging_config import logger

load_dotenv()

class Repository():

    def __init__(self,host:str="localhost"):
        self.host = host
        self.engine=None
        
    def getEngine(self) -> Engine:
        """
        This funtion gets the connects to the database and returns an engine
        """
        url = URL.create(
            "postgresql+psycopg2",
            username=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD"),
            database=os.getenv("POSTGRES_DB"),
            host=self.host
        )
    
        self.engine = create_engine(url)

    def getConnection(self):
        if self.engine is None:
            self.engine = self.getEngine()
        return self.engine.connect()
    
    def getTable(self,tableName:str):
        conn = self.getConnection()
        table = pd.read_sql(tableName,
                           conn=conn)
        conn.close()
        return table
        

    def writeTable(self,table:pd.DataFrame,tableName:str):
        conn = self.getConnection()
        
        table.to_sql(tableName,
                     conn,
                     index=False)