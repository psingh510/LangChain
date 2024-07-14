from langchain_community.llms import GooglePalm
from langchain_google_genai import GoogleGenerativeAI

from langchain_community.utilities.sql_database import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.prompts import SemanticSimilarityExampleSelector
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import FewShotPromptTemplate
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX, _mysql_prompt
from langchain.prompts.prompt import PromptTemplate
from sqlalchemy import create_engine
import mysql.connector
from few_shots import few_shots
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv
load_dotenv()

def get_few_shot_db_chain():
    db_host = 'localhost'
    db_port = 3306
    db_user = 'root'
    db_password = 'sykPAL!2025'
    db_name = 'atliq_tshirts'

    db_uri = f"mysql+pymysql://root:sykPAL!2025@localhost:3306/atliq_tshirts"
    db = SQLDatabase.from_uri(db_uri)

    
    llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key=os.environ["GOOGLE_API_KEY"], temperature=0.1)

    embedddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')
    to_vectorize = [" ".join(cases.values()) for cases in few_shots]
    vectorstore = Chroma.from_texts(to_vectorize, embedddings, metadatas = few_shots)
    
    example_selector = SemanticSimilarityExampleSelector(vectorstore=vectorstore,
        k=2,
    )
    mysql_prompt = """You are a MySQL expert. Given an input question, first create a syntactically correct MySQL query to run, then look at the results of the query and return the answer to the input question.
    Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per MySQL. You can order the results to return the most informative data in the database.
    Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in backticks (`) to denote them as delimited identifiers.
    Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
    Pay attention to use CURDATE() function to get the current date, if the question involves "today".
    
    Use the following format:
    
    Question: Question here
    SQLQuery: Query to run with no pre-amble
    SQLResult: Result of the SQLQuery
    Answer: Final answer here
    
    No pre-amble.
    """

    example_prompt = PromptTemplate(
        input_variables=["Question", "SQLQuery", "SQLResult","Answer",],
        template="\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}",
    )

    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=mysql_prompt,
        suffix=PROMPT_SUFFIX,
        input_variables=["input", "table_info", "top_k"], #These variables are used in the prefix and suffix
    )
    # llm = GooglePalm(google_api_key="AIzaSyB-jz8lMsxCf87OWauSqv5JU3pt1zrCLGY", temperature=0.1)
    db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, prompt=few_shot_prompt)
    return db_chain

chain = get_few_shot_db_chain()
response = chain.run("give me t-shirts with price over 30 and discount more than 50%")