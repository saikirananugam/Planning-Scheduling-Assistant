# This Python class uses a generative AI model to generate pandas code snippets based on user
# questions, executes the code on a DataFrame, and explains the result in plain English.
import google.generativeai as genai
from dotenv import load_dotenv
import os

SCHEMA_PROMPT = """
You are working with a pandas DataFrame named 'df' from a supply chain system.

It contains the following columns:

Raw columns:
- Top Level Branch: Sales branch managing the order (Branch1–4)
- Top Level Sold To: Client account (7281 unique customers)
- Zone: Geographic zone (West, East, North, South)
- Region Zone: Sub-zone (Zone A–D)
- Top Level Order / Line / Shipment Number: Order identifiers
- Last Next Status: Status of the order (Delivered, On Hold, etc.)
- TL SO Alert: Text-based alert (e.g., URGENT or blank)
- ECD: Actual completion date
- Promised Delivery Date: Promised delivery date
- Line Creation: Order creation date
- Schedule Pick Date: Planned pickup date
- Top Level Type: Product category
- SRP1: Unit price
- Line Amount: Total dollar value
- SC Rep: Sales contact
- ECD Notes: Delay reason (e.g., 'COVID-19 disruption')

Derived / engineered columns:
- delay_days: ECD - Promised Delivery Date
- delay_flag: 1 if delay_days > 0, else 0.  Use this for filtering delayed orders.
- lead_time: Days from Line Creation to ECD
- pickup_lead: Days from Line Creation to Schedule Pick Date
- is_urgent: 1 if TL SO Alert is URGENT, else 0.  Use this for filtering urgent orders.
- year, month, week: From Promised Delivery Date
- total_units: Line Amount / SRP1

 For urgency or delay filters, always prefer 'is_urgent' and 'delay_flag'.

 Use only valid pandas code that runs on the DataFrame 'df'. Avoid explanations.
"""
class LLMAssistant:
    def __init__(self, df):
        load_dotenv()
        self.df = df
        genai.configure(api_key=os.getenv("Gemini_API_KEY"))
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    def generate_code_from_question(self, user_question: str) -> str:
        """
        Sends the user's natural language question to Gemini with context
        and returns a pandas code snippet.
        """
        # Step 1: Combine schema and user input
        full_prompt = f"""{SCHEMA_PROMPT}

User Question: {user_question}

Return only valid pandas code using DataFrame df. Do not explain.
"""

        # Step 2: Generate response from Gemini
        response = self.model.generate_content(full_prompt)

        # Step 3: Extract the code (cleaned)
        code = response.text.strip()

        #  Remove markdown markers like ```python
        if code.startswith("```python"):
            code = code.replace("```python", "").replace("```", "").strip()

        return code
    
    def execute_code(self, code: str):
        """
        Executes the given pandas code string using self.df as context.
        Returns the result or an error message if execution fails.
        """

        # Create a local variable dictionary where `df` is available
        local_vars = {"df": self.df.copy()}  # we use a copy to protect the original df

        # ✅ Define only SAFE built-in functions you want to allow in eval/exec
        safe_builtins = {
            "len": len,
            "sum": sum,
            "min": min,
            "max": max,
            "sorted": sorted,
            "round": round,
        }

        try:
            # If the code is a simple expression (like df.shape[0] or len(df))
            if code.strip().startswith("df") or code.strip().startswith("len"):
                # eval() is used for expressions that return a value
                result = eval(code, {"__builtins__": safe_builtins}, local_vars)
            else:
                # For multi-line code or assignments, use exec()
                exec(code, {"__builtins__": safe_builtins}, local_vars)
                # Capture a variable called 'result' if defined
                result = local_vars.get("result", None)

            return result

        except Exception as e:
            return f"❌ Error executing code: {e}"
        

    def explain_result(self, user_question: str, result) -> str:
        """
        Asks Gemini to explain the result of the executed code in plain English.
        """
        prompt = f"""
    You are a helpful assistant.

    The user asked: "{user_question}"

    The answer to their question is: {result}

    Explain this result in plain English in 1–2 sentences.
    Do not restate the question. Just give the answer in a natural tone.
    """

        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        
        except Exception as e:
            return f"❌ Error explaining result: {e}"
        
            

