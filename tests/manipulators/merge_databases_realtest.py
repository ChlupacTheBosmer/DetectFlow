from detectflow.manipulators.database_manipulator import DatabaseManipulator, merge_databases
from detectflow.config import DETECTFLOW_DIR
import os

db_1 = os.path.join(os.path.dirname(DETECTFLOW_DIR), 'tests', 'manipulators', 'merge_test', 'GR2_L1_TolUmb2.db')
db_2 = os.path.join(os.path.dirname(DETECTFLOW_DIR), 'tests', 'manipulators', 'merge_test', 'GR2_L1_TolUmb3.db')
output_db = os.path.join(os.path.dirname(DETECTFLOW_DIR), 'tests', 'manipulators', 'merge_test', 'GR2_L1_TolUmb2_3.db')

merge_databases(db_1, db_2, output_db)

db_man = DatabaseManipulator(db_1)
print(db_man.fetch_one('SELECT * FROM videos'))