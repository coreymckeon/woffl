import os
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import oracledb
import pandas as pd
from scipy import optimize
from sqlalchemy import create_engine, text
from wellprofile import WellProfile

# turn on the thick mode for oracledb
oracledb.init_oracle_client()


""""
Instead of constantly re-importing the values from oracle. Just hardcode them somewhere?
Print the list or whatever and then paste it back so you don't have to pull new data
everytime you want to play around with something to verify that it works.
"""

engine = create_engine(
    f'oracle+oracledb://:@',
    connect_args={
        'user': 'ka9612',
        'password': 'fmitb0302',
        'dsn': 'pdbfprd.world'
    })

pbu_conn = engine.connect()

os.chdir(r'C:\Users\ka9612\AppData\Roaming\DBeaverData\workspace6\General\Scripts')

with open('MPU_E-42_Survey.sql', 'r') as f:
    sql_text = f.read()

df = pd.read_sql_query(sql=text(sql_text), con=pbu_conn)

print(df)
