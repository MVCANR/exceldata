import pandas as pd
import openai
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import streamlit as st
import json
from typing import Dict, Any, List
import sqlite3
from io import StringIO

class ExcelNLQuerySystem:
    """
    A comprehensive system for natural language querying of Excel employee data
    using OpenAI API and pandas DataFrames.
    """

    def __init__(self, openai_api_key: str):
        """
        Initialize the system with OpenAI API key

        Args:
            openai_api_key (str): Your OpenAI API key
        """
        self.openai_api_key = openai_api_key
        openai.api_key = openai_api_key
        self.df = None
        self.agent = None

    def load_excel_data(self, file_path: str, sheet_name: str = None) -> pd.DataFrame:
        """
        Load employee data from Excel file

        Args:
            file_path (str): Path to Excel file
            sheet_name (str): Name of the sheet to load (optional)

        Returns:
            pd.DataFrame: Loaded employee data
        """
        try:
            # Load Excel file with openpyxl engine for better Excel support
            self.df = pd.read_excel(
                file_path, 
                sheet_name=sheet_name,
                engine='openpyxl'
            )

            # Display basic info about the loaded data
            print(f"Loaded data shape: {self.df.shape}")
            print(f"Columns: {list(self.df.columns)}")

            return self.df

        except Exception as e:
            print(f"Error loading Excel file: {str(e)}")
            return None

    def setup_langchain_agent(self):
        """
        Set up LangChain pandas dataframe agent for natural language queries
        """
        if self.df is None:
            raise ValueError("No data loaded. Please load Excel data first.")

        # Initialize ChatOpenAI model
        llm = ChatOpenAI(
            temperature=0,
            model="gpt-3.5-turbo",
            openai_api_key=self.openai_api_key
        )

        # Create pandas dataframe agent
        self.agent = create_pandas_dataframe_agent(
            llm,
            self.df,
            verbose=True,
            allow_dangerous_code=True,  # Required for agent to execute code
            agent_type="openai-functions"
        )

        print("LangChain agent initialized successfully!")

    def query_with_langchain(self, question: str) -> str:
        """
        Query the data using LangChain pandas agent

        Args:
            question (str): Natural language question

        Returns:
            str: Answer from the agent
        """
        if self.agent is None:
            raise ValueError("Agent not initialized. Call setup_langchain_agent() first.")

        try:
            response = self.agent.invoke(question)
            return response['output']
        except Exception as e:
            return f"Error processing query: {str(e)}"

    def text_to_sql_query(self, question: str) -> str:
        """
        Convert natural language to SQL query using OpenAI

        Args:
            question (str): Natural language question

        Returns:
            str: Generated SQL query
        """
        # Create schema description
        schema_description = self._generate_schema_description()

        prompt = f"""
        Given the following database schema for employee data:

        {schema_description}

        Convert this natural language question to a SQL query:
        "{question}"

        Return only the SQL query without any explanation.
        """

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a SQL expert. Convert natural language questions to SQL queries."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0
            )

            sql_query = response.choices[0].message.content.strip()
            return sql_query

        except Exception as e:
            return f"Error generating SQL: {str(e)}"

    def execute_sql_on_dataframe(self, sql_query: str) -> pd.DataFrame:
        """
        Execute SQL query on pandas DataFrame using sqlite

        Args:
            sql_query (str): SQL query to execute

        Returns:
            pd.DataFrame: Query results
        """
        try:
            # Create in-memory SQLite database
            conn = sqlite3.connect(':memory:')

            # Load DataFrame into SQLite
            self.df.to_sql('employees', conn, index=False, if_exists='replace')

            # Execute query
            result = pd.read_sql_query(sql_query, conn)

            conn.close()
            return result

        except Exception as e:
            print(f"Error executing SQL: {str(e)}")
            return pd.DataFrame()

    def direct_openai_query(self, question: str) -> str:
        """
        Query data directly using OpenAI with data context

        Args:
            question (str): Natural language question

        Returns:
            str: Answer from OpenAI
        """
        # Prepare data context (first few rows and schema)
        data_sample = self.df.head(10).to_string()
        schema_info = self._generate_schema_description()

        prompt = f"""
        You are analyzing employee data. Here's the schema and a sample of the data:

        Schema:
        {schema_info}

        Sample Data:
        {data_sample}

        Question: {question}

        Based on this employee data structure, provide a detailed answer to the question.
        If you need to perform calculations, describe what calculations would be needed.
        """

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a data analyst expert specializing in HR and employee data analysis."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"Error querying OpenAI: {str(e)}"

    def _generate_schema_description(self) -> str:
        """
        Generate a description of the DataFrame schema for AI context

        Returns:
            str: Schema description
        """
        if self.df is None:
            return "No data loaded"

        schema_parts = []
        schema_parts.append(f"Table: employees ({self.df.shape[0]} rows)")
        schema_parts.append("Columns:")

        for col in self.df.columns:
            dtype = str(self.df[col].dtype)
            null_count = self.df[col].isnull().sum()
            unique_count = self.df[col].nunique()

            schema_parts.append(f"  - {col} ({dtype}): {unique_count} unique values, {null_count} nulls")

        return "\n".join(schema_parts)

    def create_sample_employee_data(self) -> pd.DataFrame:
        """
        Create sample employee data for demonstration

        Returns:
            pd.DataFrame: Sample employee data
        """
        import random
        from datetime import datetime, timedelta

        # Sample data generation
        first_names = ['John', 'Jane', 'Michael', 'Sarah', 'David', 'Lisa', 'Robert', 'Emily', 'James', 'Anna']
        last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez']
        departments = ['Engineering', 'Sales', 'Marketing', 'HR', 'Finance', 'Operations']
        positions = ['Manager', 'Senior', 'Associate', 'Analyst', 'Director', 'VP']
        countries = ['USA', 'Canada', 'UK', 'Germany', 'France', 'Australia']
        states = ['CA', 'NY', 'TX', 'FL', 'WA', 'IL', 'PA', 'OH', 'GA', 'NC']

        data = []
        manager_ids = []

        # Generate 100 sample employees
        for i in range(100):
            emp_id = f"EMP{i+1:04d}"
            first_name = random.choice(first_names)
            last_name = random.choice(last_names)

            # Create hierarchical structure
            if i < 5:  # Top level managers
                manager_id = None
                level_2_manager = None
                level_3_manager = None
                level_4_manager = None
                level_5_manager = None
                manager_ids.append(emp_id)
            elif i < 20:  # Second level
                manager_id = random.choice(manager_ids[:5]) if manager_ids else None
                level_2_manager = None
                level_3_manager = None
                level_4_manager = None
                level_5_manager = None
            else:  # Lower levels
                manager_id = random.choice(manager_ids) if manager_ids else None
                level_2_manager = random.choice(manager_ids[:5]) if len(manager_ids) >= 5 else None
                level_3_manager = random.choice(manager_ids[:10]) if len(manager_ids) >= 10 else None
                level_4_manager = random.choice(manager_ids[:15]) if len(manager_ids) >= 15 else None
                level_5_manager = None

            employee = {
                'employee_id': emp_id,
                'first_name': first_name,
                'last_name': last_name,
                'full_name': f"{first_name} {last_name}",
                'manager_id': manager_id,
                'level_2_manager': level_2_manager,
                'level_3_manager': level_3_manager,
                'level_4_manager': level_4_manager,
                'level_5_manager': level_5_manager,
                'department': random.choice(departments),
                'position': random.choice(positions),
                'country': random.choice(countries),
                'state': random.choice(states),
                'city': f"City_{random.randint(1, 50)}",
                'salary': random.randint(40000, 150000),
                'hire_date': datetime.now() - timedelta(days=random.randint(30, 2000)),
                'status': random.choice(['Active', 'Active', 'Active', 'Active', 'Inactive'])
            }

            data.append(employee)

            # Add some employees as potential managers
            if random.random() < 0.3 and len(manager_ids) < 30:
                manager_ids.append(emp_id)

        self.df = pd.DataFrame(data)
        return self.df

# Example usage functions
def example_basic_usage():
    """
    Example of basic usage with the ExcelNLQuerySystem
    """
    # Initialize the system (replace with your actual OpenAI API key)
    system = ExcelNLQuerySystem("your-openai-api-key-here")

    # Create sample data for demonstration
    df = system.create_sample_employee_data()
    print("Sample data created successfully!")
    print(df.head())

    # Set up LangChain agent
    # system.setup_langchain_agent()

    # Example queries
    questions = [
        "How many employees are in each department?",
        "Who are the managers in the Engineering department?",
        "What is the average salary by department?",
        "Which employees report to manager EMP0001?",
        "How many employees are located in each country?"
    ]

    # Query using direct OpenAI approach
    for question in questions:
        print(f"\nQuestion: {question}")
        answer = system.direct_openai_query(question)
        print(f"Answer: {answer}")

# Streamlit app example
def create_streamlit_app():
    """
    Create a Streamlit web application for the Excel NL Query system
    """
    streamlit_code = """
import streamlit as st
import pandas as pd
from excel_nl_query_system import ExcelNLQuerySystem

def main():
    st.title("Excel Employee Data - Natural Language Query System")
    st.sidebar.header("Configuration")

    # API key input
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")

    if not api_key:
        st.warning("Please enter your OpenAI API key in the sidebar.")
        return

    # Initialize system
    @st.cache_resource
    def init_system(api_key):
        return ExcelNLQuerySystem(api_key)

    system = init_system(api_key)

    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload Excel file", type=['xlsx', 'xls'])

    if uploaded_file:
        # Load data
        with st.spinner("Loading Excel data..."):
            df = pd.read_excel(uploaded_file, engine='openpyxl')
            system.df = df

        st.success(f"Data loaded successfully! Shape: {df.shape}")

        # Display data preview
        with st.expander("Data Preview"):
            st.dataframe(df.head(10))

        # Query interface
        st.header("Ask Questions About Your Data")

        # Predefined questions
        predefined_questions = [
            "How many employees are in each department?",
            "What is the average salary by department?",
            "Who are the top 5 highest paid employees?",
            "How many employees report to each manager?",
            "What is the distribution of employees by country?"
        ]

        selected_question = st.selectbox("Select a predefined question:", [""] + predefined_questions)

        # Custom question input
        custom_question = st.text_area("Or ask your own question:", height=100)

        question = selected_question if selected_question else custom_question

        if st.button("Get Answer") and question:
            with st.spinner("Processing your question..."):
                # Choose query method
                method = st.sidebar.radio("Query Method", ["Direct OpenAI", "Text-to-SQL"])

                if method == "Direct OpenAI":
                    answer = system.direct_openai_query(question)
                    st.markdown("### Answer:")
                    st.write(answer)

                elif method == "Text-to-SQL":
                    sql_query = system.text_to_sql_query(question)
                    st.markdown("### Generated SQL:")
                    st.code(sql_query, language="sql")

                    if st.button("Execute SQL"):
                        result = system.execute_sql_on_dataframe(sql_query)
                        if not result.empty:
                            st.markdown("### Results:")
                            st.dataframe(result)
                        else:
                            st.error("No results or error in query execution.")

    else:
        # Demo with sample data
        if st.sidebar.button("Use Sample Data"):
            with st.spinner("Creating sample employee data..."):
                df = system.create_sample_employee_data()

            st.success(f"Sample data created! Shape: {df.shape}")

            with st.expander("Sample Data Preview"):
                st.dataframe(df.head(10))

if __name__ == "__main__":
    main()
"""

    return streamlit_code

# Save the main solution to a file
with open('excel_nl_query_system.py', 'w') as f:
    f.write(solution_code)

print("Solution code saved to 'excel_nl_query_system.py'")

# Also save the Streamlit app
streamlit_app_code = create_streamlit_app()
with open('streamlit_app.py', 'w') as f:
    f.write(streamlit_app_code)

print("Streamlit app code saved to 'streamlit_app.py'")
