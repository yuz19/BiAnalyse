import mysql.connector

class TableFait:
    def __init__(self, conn):
        self.conn = conn

    # Fonction pour vérifier si une colonne est de type numérique
    def is_numeric_column(self, column_type):
        return 'int' in column_type or 'float' in column_type or 'double' in column_type

    # Fonction pour vérifier si une table est une table de faits
    def is_fact_table(self, table_name):
        cursor = self.conn.cursor()
        cursor.execute(f"SHOW COLUMNS FROM {table_name}")
        columns = cursor.fetchall()
        fact_columns = []
        for col in columns:
            column_name = col[0]
            column_type = col[1].lower()
            
            print(column_name,":",column_type,"\n")
            
            if 'pri' not in col[3].lower() and 'foreign' not in col[3].lower():
                if not self.is_numeric_column(column_type):
                    return False, []
                else:
                    fact_columns.append(column_name)
        return True, fact_columns

    # Fonction pour obtenir les tables de faits
    def get_mesure(self):
        cursor = self.conn.cursor()
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        
        print(tables)
        
        fact_tables = {}
        for table in tables:
            table_name = table[0]
            is_fact, columns = self.is_fact_table(table_name)
            if is_fact:
                fact_tables[table_name] = columns
            print("fin table")
        return fact_tables
 