# The code you provided is using a Python script to interact with an `LLMAssistant` object that is
# designed to assist with analyzing and processing data. Here's a breakdown of what the code is doing:
from src.llm import LLMAssistant
from src.ingest import DataIngestor

loader = DataIngestor("data/sales_data.csv")
df = loader.load_and_clean()

assistant = LLMAssistant(df)

# question = "How many urgent orders were delayed in 2023?"
# code = assistant.generate_code_from_question(question)

# print("🔍 Gemini generated this code:")
# print(code)

question="How many years of data do we have"
code = assistant.generate_code_from_question(question)
print(" Code:", code)

result = assistant.execute_code(code)
print("Result:", result)

explanation = assistant.explain_result(question, result)
print(" Explanation:", explanation)