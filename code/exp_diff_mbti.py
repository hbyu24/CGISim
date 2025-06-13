# %%
import concurrent.futures
import datetime
import json
import random
import warnings
import re
import numpy as np

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import associative_memory
from concordia.associative_memory import formative_memories
from concordia.associative_memory import importance_function
from concordia.clocks import game_clock
from concordia.components import agent as agent_components
from concordia.components.agent import question_of_recent_memories
from concordia.memory_bank import legacy_associative_memory
from concordia.typing import entity
from concordia.utils import measurements as measurements_lib
import time
warnings.filterwarnings('ignore', category=FutureWarning)
import yaml
from save_simulation_result import SimulationResultSaver
# %% md
# # model and embedder
# %%
import sentence_transformers
from concordia.language_model import utils
from concordia.language_model import azure_gpt_model
from concordia.language_model import no_language_model
from concordia.language_model import gpt_model
from concordia.language_model import xai_grok_model
from concordia.language_model import qwen3_model
import os


class BasicUtils:
    def __init__(self, model_names, disable_language_model=False, primary_model_index=0):
        """
        Initializes the utility class by setting up the embedder and specified language models.
        """
        self.embedder = self.embedder_setup()
        self.model_list = {}

        # Handle both single string and list inputs for model names
        if isinstance(model_names, str):
            model_names = [model_names]

        # Initialize each model and store it in the model_list dictionary
        for model_name in model_names:
            print(f"Initializing model: {model_name}")
            self.model_list[model_name] = self.model_setup(model_name, disable_language_model)

        # Set the primary model for easy access
        if model_names:
            primary_name = model_names[primary_model_index] if primary_model_index < len(model_names) else model_names[
                0]
            self.model = self.model_list[primary_name]
        else:
            self.model = None

    def embedder_setup(self):
        """
        Sets up the sentence embedder using a standard public model.
        """
        st_model = sentence_transformers.SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        embedder = lambda x: st_model.encode(x, show_progress_bar=False)
        return embedder

    def get_model(self, model_name=None):
        """
        Retrieves a specific model by its name, or the primary model if no name is specified.
        """
        if model_name is None:
            return self.model
        elif model_name in self.model_list:
            return self.model_list[model_name]
        else:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.model_list.keys())}")

    def model_setup(self, model_name, disable_language_model=False):
        """
        Sets up a language model based on its name, loading credentials from environment variables.
        """
        model = None

        # To debug without incurring API costs, set disable_language_model=True.
        if disable_language_model:
            # return no_language_model.NoLanguageModel() # Uncomment if you have this module
            print("Language model is disabled.")
            return None

        # --- OpenAI & compatible models ---
        if model_name in ['gpt-4o', 'gpt-4o-mini', 'deepseek-v3', 'deepseek-r1', 'gpt-4.1', 'gpt-4.1-mini',
                          'gpt-4.1-nano']:
            # All these models use the OpenAI API format.
            # The exact model version string (e.g., 'gpt-4o-2024-11-20') can be passed directly.
            model = utils.language_model_setup(
                api_type='openai',
                model_name=model_name,  # Pass the generic name, let the underlying function handle specifics if needed
                api_key=os.environ.get('OPENAI_API_KEY'),
                disable_language_model=disable_language_model,
            )

        # --- Azure OpenAI models ---
        elif model_name.startswith('azure-'):
            # Example model_name: 'azure-gpt-4o'
            # The actual deployment name (e.g., 'gpt-4o-2024-11-20') should be handled in AzureGptLanguageModel
            deployment_name = model_name.replace('azure-', '')  # A simple way to get the deployment name
            model = azure_gpt_model.AzureGptLanguageModel(
                model_name=deployment_name,  # Or pass the full specific name if your class requires it
                api_key=os.environ.get('AZURE_OPENAI_API_KEY'),
                azure_endpoint=os.environ.get('AZURE_OPENAI_ENDPOINT'),
                api_version=os.environ.get('AZURE_OPENAI_API_VERSION'),  # Use default if not set
            )

        # --- XAI Grok models ---
        elif model_name.startswith('grok-'):
            # Example model_name: 'grok-3' or 'grok-3-mini'
            # The actual model name for the API might have a '-beta' suffix
            api_model_name = f"{model_name}-beta" if 'beta' not in model_name else model_name
            model = xai_grok_model.GrokLanguageModel(
                model_name=api_model_name,
                api_key=os.environ.get('XAI_API_KEY')
            )

        # --- Qwen (Dashscope) models ---
        elif model_name.startswith('qwen'):
            dashscope_api_key = os.environ.get('DASHSCOPE_API_KEY')
            if not dashscope_api_key:
                print("Warning: DASHSCOPE_API_KEY environment variable not found.")

            model = qwen3_model.Qwen3LanguageModel(
                model_name=model_name,
                api_key=dashscope_api_key,
                enable_thinking=False
            )

        else:
            raise ValueError(f"Unknown model name: {model_name}")

        return model

disable_language_model= False
Model_And_Embedder = BasicUtils(['gpt-4.1', 'gpt-4.1-mini', 'gpt-4.1-nano'],disable_language_model)
model = Model_And_Embedder.model_list['gpt-4.1-mini']
model_high=Model_And_Embedder.model_list['gpt-4.1']
model_low=Model_And_Embedder.model_list['gpt-4.1-nano']
embedder = Model_And_Embedder.embedder
importance_model = importance_function.AgentImportanceModel(model_low)
importance_model_gm = importance_function.ConstantImportanceModel(model_low)

def extract_fields_with_model(response, format_example, fields_to_extract, model=model_low):
    """
    Use language model to extract structured fields from a response after regex matching has failed.

    Args:
        response (str): The model's text response
        format_example (str): Example format string showing expected structure
        fields_to_extract (list): List of field names to extract
        model (LanguageModel, optional): Model to use for extraction

    Returns:
        dict: Dictionary mapping field names to extracted values
    """
    # Use provided model or fall back to a default if needed
    if model is None:
        # This function needs a model passed to it, since it's not a class method
        raise ValueError("A language model must be provided")

    # Create a clean prompt that asks for pure extraction
    prompt = f"""I need to extract structured information from this response.
The response should follow this format:
"{format_example}"

But the response I received is:
"{response}"

Please extract the following fields exactly as they appear:
{", ".join(fields_to_extract)}

Format your answer as a simple dictionary with no explanation:
{{
    "field_name": "extracted_value",
    ...
}}"""

    try:
        # Get extraction from model
        extraction_result = model.sample_text(
            prompt=prompt,
            # max_tokens=300,
            temperature=0.1  # Low temperature for consistent extraction
        )

        # Process the result into a dictionary
        result_dict = {}

        # Try to parse as proper Python dictionary if it contains one
        import re
        import ast

        # Look for dictionary-like structure
        dict_match = re.search(r'\{[^}]+\}', extraction_result, re.DOTALL)
        if dict_match:
            try:
                # Try to parse the extracted dictionary
                result_dict = ast.literal_eval(dict_match.group(0))
            except (SyntaxError, ValueError):
                # If parsing fails, extract fields individually
                pass

        # If dictionary parsing failed, extract each field individually
        if not result_dict:
            for field in fields_to_extract:
                field_pattern = rf'"{re.escape(field)}":\s*"([^"]+)"'
                match = re.search(field_pattern, extraction_result)
                if match:
                    result_dict[field] = match.group(1)
                else:
                    # Try without quotes around value
                    field_pattern = rf'"{re.escape(field)}":\s*([^,\n}}]+)'
                    match = re.search(field_pattern, extraction_result)
                    if match:
                        result_dict[field] = match.group(1).strip()
                    else:
                        result_dict[field] = None

        # Ensure all requested fields are in the result
        for field in fields_to_extract:
            if field not in result_dict:
                result_dict[field] = None

        return result_dict

    except Exception as e:
        print(f"Error using model to extract fields: {e}")
        # Return empty values as fallback
        return {field: None for field in fields_to_extract}


def extract_confidence_distribution(response_text, options, model=model_low):

    # Get the list of option keys
    option_keys = list(options.keys()) if isinstance(options, dict) else options

    # Create prompt for reliable extraction
    extraction_prompt = f"""Extract the confidence distribution from the following response text.

The response should contain confidence percentages for each option. Please extract these and convert them to decimal format (0-1 scale).

Available options: {', '.join(option_keys)}

Response text to extract from:
"{response_text}"

Instructions:
1. Find each option and its confidence percentage in the response
2. Convert percentages to decimal format (e.g., 75% becomes 0.75)
3. Ensure all confidence values sum to approximately 1.0 (allow slight rounding differences)
4. Provide the result in a clean JSON format

Return ONLY a JSON object in this format:
{{
    "Option A": 0.0,
    "Option B": 0.0,
    "Option C": 0.0,
    "Option D": 0.0
}}

If any option is not mentioned, assume it has 0% confidence.
"""

    try:
        # Get extraction from model
        extraction_result = model.sample_text(
            prompt=extraction_prompt,
            max_tokens=300,
            temperature=0.1  # Low temperature for consistent extraction
        )

        # Parse the result as JSON
        import json
        import re

        # Find JSON object in the response
        json_match = re.search(r'\{[^}]+\}', extraction_result, re.DOTALL)
        if json_match:
            try:
                confidence_dict = json.loads(json_match.group(0))

                # Validate and clean the result
                cleaned_dict = {}
                for key, value in confidence_dict.items():
                    # Find matching option key (case-insensitive)
                    matching_key = None
                    for option_key in option_keys:
                        if key.lower().replace(" ", "") == option_key.lower().replace(" ", ""):
                            matching_key = option_key
                            break

                    if matching_key:
                        # Ensure value is float and in range [0, 1]
                        try:
                            float_value = float(value)
                            # Convert percentage to decimal if needed
                            if float_value > 1:
                                float_value = float_value / 100
                            cleaned_dict[matching_key] = max(0.0, min(1.0, float_value))
                        except ValueError:
                            cleaned_dict[matching_key] = 0.0

                # Ensure all options are present
                for option_key in option_keys:
                    if option_key not in cleaned_dict:
                        cleaned_dict[option_key] = 0.0

                # Normalize to sum to 1.0
                total = sum(cleaned_dict.values())
                if total > 0:
                    normalized_dict = {k: v / total for k, v in cleaned_dict.items()}

                    # Verify normalization worked
                    if abs(sum(normalized_dict.values()) - 1.0) < 0.0001:
                        return normalized_dict

            except json.JSONDecodeError:
                pass

        # If JSON parsing failed, try a more structured prompt
        fallback_prompt = f"""The previous extraction attempt failed. Please try again with this specific format:

From this text:
"{response_text}"

Extract the confidence percentages for each option and format them exactly like this:

Option A: 0.XX
Option B: 0.XX
Option C: 0.XX
Option D: 0.XX

Where XX are decimal numbers between 0 and 1, representing the confidence percentages.
All values must sum to 1.0. If an option is not mentioned, use 0.00.
"""

        fallback_result = model.sample_text(
            prompt=fallback_prompt,
            max_tokens=200,
            temperature=0.1
        )

        # Parse the fallback result
        result_dict = {}
        for option_key in option_keys:
            pattern = rf"{re.escape(option_key)}:\s*(0?\.\d+|1\.0+)"
            match = re.search(pattern, fallback_result, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                    result_dict[option_key] = max(0.0, min(1.0, value))
                except ValueError:
                    result_dict[option_key] = 0.0
            else:
                result_dict[option_key] = 0.0

        # Normalize if needed
        total = sum(result_dict.values())
        if total > 0:
            result_dict = {k: v / total for k, v in result_dict.items()}
        else:
            # If all values are 0, distribute evenly
            even_value = 1.0 / len(option_keys)
            result_dict = {k: even_value for k in option_keys}

        return result_dict

    except Exception as e:
        print(f"Error in extract_confidence_distribution: {e}")
        # As last resort, distribute confidence evenly
        even_confidence = 1.0 / len(option_keys)
        return {k: even_confidence for k in option_keys}


# %% md
# # Background Info
# %%
class BackgroundInfo:
    def __init__(self, agent_num, ceo_name):
        self.instructions_CEO = """You and {agent_num} other people have invested together to start an investment company. You are the CEO of the company. Everyone has an equal share in the company. The company starts with 100k, which will be used for investments across five years. Each year, the shareholders and the CEO will make strategic investment decisions to grow the company’s capital by investing in one asset per quarter—cash savings, bonds, real estate, or stocks. These assets have unique risks and returns, which are influenced by varying market conditions. At the start of each year, the CEO will hold a shareholders' meeting to decide the annual investment budget. Then, at the beginning of each quarter, the CEO will hold a meeting to decide which asset to invest in. At the end of the year, the CEO will hold a meeting to review the company’s performance and for shareholders to evaluate the CEO’s leadership.""".format(
            agent_num=agent_num, ceo_name=ceo_name)
        self.instructions_Shareholder = """You and {agent_num} other people have invested together to start an investment company. You are a shareholder of the company and {ceo_name} is CEO. Everyone has an equal share in the company. The company starts with 100k, which will be used for investments across five years. Each year, the shareholders and the CEO will make strategic investment decisions to grow the company’s capital by investing in one asset per quarter—cash savings, bonds, real estate, or stocks. These assets have unique risks and returns, which are influenced by varying market conditions. At the start of each year, the CEO will hold a shareholders' meeting to decide the annual investment budget. Then, at the beginning of each quarter, the CEO will hold a meeting to decide which asset to invest in. At the end of the year, the CEO will hold a meeting to review the company’s performance and for shareholders to evaluate the CEO’s leadership.""".format(
            agent_num=agent_num, ceo_name=ceo_name)
        self.annual_budget_meeting = """**Annual Budget Meeting (Start of the Year):**
This meeting opens with the CEO presenting the market condition for the year (stable, expansion, high inflation, or recession), reviewing the previous year's performance, and sharing historical asset return data. The CEO then introduces four budget allocation options (Options A-D) for the year.

The meeting follows a structured open discussion format where:
1. The CEO leads multiple rounds of discussion, with shareholders taking turns to express their views
2. Each shareholder can speak or choose to remain silent when their turn comes
3. The CEO may respond to individual comments to guide the discussion
4. After sufficient discussion, the CEO selects one budget option as the formal proposal

The proposal is then put to a shareholder vote, requiring a 2/3 majority to pass. If the proposal fails to achieve the required majority, the discussion process resumes until a consensus is reached and an approved proposal is secured."""
        self.quarterly_investment_meetings = """**Quarterly Investment Meetings (Q1 to Q4):**
At the beginning of each quarter, the CEO presents the previous quarter's investment results, historical asset performance data, and the current quarter's investment options (cash savings, bonds, real estate, stocks) with their risk-return profiles under the current market conditions.

Similar to the annual budget meeting, these meetings use a structured open discussion format:
1. The CEO guides multiple rounds of focused discussion on the four asset options
2. Shareholders speak in turn, expressing preferences and reasoning in 1-2 concise sentences
3. The CEO may respond to individual comments to facilitate decision-making
4. Based on the discussion, the CEO selects one asset as the formal investment proposal

The selected asset proposal requires a 2/3 majority vote to be approved. If not achieved, the discussion process continues until a consensus is reached. Once approved, the allocated funds for that quarter are invested in the chosen asset. At quarter's end, the investment is automatically liquidated to calculate returns, which affects the company's total capital."""
        self.annual_review_meeting = """**Annual Review Meeting (End of the Year):**
This meeting uses a written submission format rather than open discussion. The CEO begins by presenting a comprehensive summary of the year's financial performance, including investment decisions made, returns achieved, and overall capital growth.

Following this presentation, shareholders individually submit written evaluations covering:
1. Their assessment of the company's performance during the year
2. Their evaluation of the CEO's leadership effectiveness
3. Numerical ratings (1-10 scale) for both company performance and CEO leadership

After reviewing all submissions, the CEO provides a response addressing key themes from the feedback and outlining improvement plans for the following year. The meeting concludes with the CEO summarizing the average evaluation scores and setting expectations for the upcoming year."""
        self.investment_detail = """**Investment Details:**
    The company’s initial 100k is used for investments. Each quarter, the selected asset’s returns (or losses) adjust the company’s capital based on the market condition. The assets have the following general characteristics:
    - **Cash Savings:** No risk, but the lowest return.
    - **Bonds:** Low risk, low return.
    - **Real Estate:** Medium risk, medium return.
    - **Stocks:** High risk, high return.
    These characteristics can change based on the market conditions, which vary each year. """
        self.role_ceo = """You are the CEO of an investment company jointly owned by you and the other shareholders. Your primary responsibility is to lead meetings effectively and make strategic investment decisions that will grow the company's capital.

As CEO, you will:
1. Lead three types of meetings: annual budget meetings, quarterly investment meetings, and annual review meetings.
2. For investment-related meetings (annual budget and quarterly investments), you'll use open discussions where you'll guide conversations toward consensus by:
   - Starting with a clear opening statement explaining the market conditions and available options
   - Planning structured discussion rounds with specific topics
   - Responding to shareholders' comments strategically
   - Summarizing discussions and identifying trends in opinions
   - Ultimately selecting a proposal you believe is best for the company

3. For the annual review meeting, you'll collect written feedback from shareholders about your leadership and company performance.
4. For all decisions, you must persuade shareholders to accept your proposals, requiring a 2/3 majority vote to proceed.

Throughout all interactions, your leadership style should authentically reflect your MBTI personality traits. This includes how you organize discussions, respond to shareholder input, handle disagreements, and make final decisions. Your goal is to balance maximizing returns with maintaining positive shareholder relations. """
        self.role_shareholder = """You are a shareholder with equal ownership in this investment company. Your role is to actively participate in company governance and help make sound investment decisions that align with your personal judgment.

As a shareholder, you will:
1. Participate in three types of meetings: annual budget meetings (deciding yearly investment allocations), quarterly investment meetings (selecting specific assets), and annual review meetings (evaluating performance).

2. During investment-related meetings (annual budget and quarterly), you'll engage in open discussions where you can:
   - Choose whether to speak when your turn comes (based on your personality and interest in the topic)
   - Express your opinions on proposed options in 1-2 concise sentences
   - Ask questions or respond to the CEO's comments
   - Vote on the CEO's final proposal (requiring 2/3 majority to pass)

3. For annual review meetings, you'll submit written feedback on the CEO's leadership and the company's performance, then provide numerical ratings (1-10 scale) for both aspects.

Your decisions should authentically reflect your MBTI personality type - whether you're analytical, emotional, cautious, or bold in your approach. Consider how your personality would naturally respond to investment opportunities, risk levels, and interpersonal dynamics in these meetings. """

        # Add market condition descriptions
        self.market_condition_intro = {}
        self.market_condition_intro[
            'stable'] = """A balanced market environment characterized by steady economic growth, consistent interest rates, and low volatility across all quarters. In this predictable landscape, stocks offer balanced potential for both gains and losses with moderate volatility, while real estate tends to maintain steady appreciation with occasional flat periods. Bonds generally deliver consistent returns with minimal risk, though they yield less than growth-oriented assets in favorable conditions. Cash savings provide complete safety but offer the lowest returns, serving primarily for capital preservation rather than growth. The minimal seasonal variations create relatively uniform investment opportunities throughout the year."""

        self.market_condition_intro[
            'expansion'] = """A growth-oriented market with accelerating economic momentum that often builds throughout the year, driven by rising business prosperity, increased consumer spending, and favorable lending conditions. Stocks typically exhibit stronger upward potential with reduced likelihood of significant losses (though corrections remain possible), while real estate benefits from increased demand, showing more consistent appreciation than in other periods. Bonds maintain stability but generally underperform relative to growth assets in this environment. Cash represents the safest option but typically means missing the higher returns available elsewhere. The building momentum often creates enhanced opportunities in later quarters."""

        self.market_condition_intro[
            'high_inflation'] = """An economic environment where rapidly rising prices erode purchasing power, creating a dynamic investment landscape as different sectors respond variably to inflationary pressures. Stocks can serve as partial inflation hedges, potentially maintaining value in real terms despite increased volatility, while real estate historically performs as an inflation hedge, though with greater market fluctuations. Bonds face significant challenges as inflation erodes their fixed returns, potentially resulting in negative real yields despite nominal gains. Cash savings are most vulnerable to inflation's effects, often losing purchasing power despite nominal stability. Central bank responses typically trigger interest rate adjustments throughout the year, affecting asset valuations differently across quarters."""

        self.market_condition_intro[
            'recession'] = """A contracting economic environment with declining indicators and reduced business activity, potentially varying in severity across quarters with early periods often experiencing sharper declines. Stocks face considerable headwinds with increased probability of losses, though selective opportunities may exist at discounted valuations. Real estate typically suffers from reduced liquidity and potential value declines, presenting both risks and possible long-term buying opportunities as the year progresses. Bonds frequently emerge as relative safe havens, offering higher yields with manageable risk, while cash provides maximum capital preservation when other assets decline. Market sentiment fluctuates between pessimism and cautious optimism as economic indicators evolve, potentially creating strategic opportunities at different points during the year."""


# %% md
# # generate profiles
# %%
def escape_curly_braces(text: str) -> str:
    """Escapes curly braces in a string to avoid formatting issues."""
    return text.replace("{", "{{").replace("}", "}}")

def ChooseProfile(num_agents, specify_mbti=False, mbti_specs=None, distinct_mbti=False, random_seed=None):
    """
    Selects a specified number of agent profiles from a data bank based on MBTI criteria.

    Args:
        num_agents (int): The total number of agent profiles to select.
        specify_mbti (bool): If True, profiles are selected based on the 'mbti_specs' argument.
        mbti_specs (list or dict, optional): The specific MBTI criteria.
            - If a list (e.g., ['INTJ', 'ENFP']): Selects from profiles with these MBTI types.
            - If a dict (e.g., {'INTJ': 2, 'ENFP': 3}): Selects a specific number of profiles for each MBTI type.
            Defaults to None.
        distinct_mbti (bool): If True, ensures that all selected agents have unique MBTI types.
        random_seed (int, optional): A seed for the random number generator for reproducibility. Defaults to None.

    Returns:
        list: A list of selected profile dictionaries.
        int: Returns -1 if the input parameters are invalid.
    """
    root_dir = os.getcwd()
    try:
        with open(os.path.join(root_dir, 'data', 'mbti_1024_bank_new.json'), 'r', encoding='utf-8') as f:
            data_list = [json.loads(line) for line in f]
    except FileNotFoundError:
        print("Error: The data file 'data/mbti_1024_bank_new.json' was not found.")
        return -1

    mbti_types = [
        "ISTJ", "ISFJ", "INFJ", "INTJ", "ISTP", "ISFP", "INFP", "INTP",
        "ESTP", "ESFP", "ENFP", "ENTP", "ESTJ", "ESFJ", "ENFJ", "ENTJ"
    ]
    selected_profiles = []

    if random_seed is not None:
        random.seed(random_seed)

    if specify_mbti:
        if mbti_specs is None:
            print("Error: Please provide MBTI specifications when 'specify_mbti' is True.")
            return -1

        if isinstance(mbti_specs, list):
            if distinct_mbti and (num_agents > len(mbti_specs)):
                print("Error: The number of specified MBTI types must be >= num_agents when 'distinct_mbti' is True.")
                return -1

            filtered_data_all = []
            if distinct_mbti:
                for mbti_type in mbti_specs:
                    filtered_data = [item for item in data_list if item['mbti'] == mbti_type]
                    if filtered_data:
                        # Randomly select one element
                        random_item = random.choice(filtered_data)
                        filtered_data_all.append(random_item)
                    else:
                        print(f"Warning: No instance found for MBTI type '{mbti_type}'.")
                selected_profiles = random.sample(filtered_data_all, num_agents)
            else:
                for mbti_type in mbti_specs:
                    # Filter dictionaries with the specified mbti type
                    filtered_data = [item for item in data_list if item['mbti'] == mbti_type]
                    # Ensure the filtered list is not empty
                    if filtered_data:
                        filtered_data_all.extend(filtered_data)
                    else:
                        print(f"Warning: No instance found for MBTI type '{mbti_type}'.")
                if len(filtered_data_all) < num_agents:
                    print(
                        "Error: Not enough profiles available for the specified MBTI types to meet the required number of agents.")
                    return -1
                selected_profiles = random.sample(filtered_data_all, num_agents)

            return selected_profiles

        elif isinstance(mbti_specs, dict):
            if sum(mbti_specs.values()) != num_agents:
                print("Error: The sum of values in the mbti_specs dictionary must equal num_agents.")
                return -1

            for mbti_type, count in mbti_specs.items():
                # Filter dictionaries with the specified mbti type
                filtered_data = [item for item in data_list if item['mbti'] == mbti_type]
                # Ensure the filtered list is not empty
                if len(filtered_data) >= count:
                    selected_profiles.extend(random.sample(filtered_data, count))
                else:
                    print(
                        f"Warning: Not enough instances for MBTI type '{mbti_type}'. Required: {count}, Found: {len(filtered_data)}.")
            return selected_profiles

    else:  # Not specifying MBTI, general selection
        if distinct_mbti and (num_agents > 16):
            print("Error: Number of agents cannot exceed 16 when 'distinct_mbti' is True.")
            return -1

        if distinct_mbti:
            mbti_selected = random.sample(mbti_types, num_agents)
            for mbti_type in mbti_selected:
                # Filter dictionaries with the specified mbti type
                filtered_data = [item for item in data_list if item['mbti'] == mbti_type]
                # Ensure the filtered list is not empty
                if filtered_data:
                    # Randomly select one element
                    selected_profiles.append(random.choice(filtered_data))
                else:
                    print(f"Warning: No instance found for MBTI type '{mbti_type}'.")
        else:
            selected_profiles = random.sample(data_list, num_agents)

        return selected_profiles


def SetupLeader(selected_profiles, leader_mbti=None, random_seed=None):
    """
    Selects a leader from the given profiles, either randomly or by a specified MBTI type.

    Args:
        selected_profiles (list): A list of profile dictionaries from which to choose a leader.
        leader_mbti (list, optional): A list of desired MBTI types for the leader.
                                     If None, a leader is chosen randomly. Defaults to None.
        random_seed (int, optional): An integer seed for reproducible random selection. Defaults to None.

    Returns:
        dict: A dictionary representing the selected leader's profile.
        int: Returns -1 if no suitable leader with the specified MBTI is found.
    """
    if random_seed is not None:
        random.seed(random_seed)

    if not selected_profiles:
        print("Error: The list of selected profiles is empty.")
        return -1

    if leader_mbti:
        # Filter profiles that match the desired leader MBTI types
        eligible_leaders = [p for p in selected_profiles if p['mbti'] in leader_mbti]

        if eligible_leaders:
            leader = random.choice(eligible_leaders)
            return leader
        else:
            print("Warning: No profile with the specified MBTI types found among selected profiles.")
            # Fallback to random selection among all profiles
            leader = random.choice(selected_profiles)
            return leader
    else:
        # Select a leader randomly from all profiles
        leader = random.choice(selected_profiles)
        return leader

# %%
def setup_clock(start_year=1971):
    # Start on January 1st of the start year at 8:00 AM
    start_time = datetime.datetime(year=start_year, month=1, day=1, hour=8)

    # Major time step for meetings (5 minutes)
    major_time_setup = datetime.timedelta(minutes=5)

    # Minor time step for quick updates
    minor_time_setup = datetime.timedelta(seconds=10)

    # Create the clock with both time steps
    clock = game_clock.MultiIntervalClock(
        start=start_time,
        step_sizes=[major_time_setup, minor_time_setup])

    return clock


# %% md
# # Market
# %%
class Market:
    """Manages market conditions and asset returns for the simulation."""

    def __init__(self, seed=None):

        if seed is not None:
            random.seed(seed)

        # Define the market conditions for 5 years (20 quarters)
        self.market_conditions = ['stable', 'expansion', 'high_inflation', 'recession', 'stable']
        self.assets = ['cash', 'bonds', 'real_estate', 'stocks']

        # Define return distributions for each asset under each market condition
        self.asset_distributions = {
            'stable': {
                'cash': 0.02,
                'bonds': {'rate': 0.05, 'default_prob': 0.05},
                'real_estate': {'returns': [0.10, 0.00, -0.05], 'probs': [0.7, 0.2, 0.1]},
                'stocks': {'returns': [0.20, -0.10], 'probs': [0.5, 0.5]}
            },
            'expansion': {
                'cash': 0.02,
                'bonds': {'rate': 0.05, 'default_prob': 0.05},
                'real_estate': {'returns': [0.10, 0.00], 'probs': [0.8, 0.2]},
                'stocks': {'returns': [0.15, 0.00], 'probs': [0.7, 0.3]}
            },
            'high_inflation': {
                'cash': lambda: random.uniform(-0.01, 0.02),  # Simplified from range -1% to 2% to a fixed 0.5%
                'bonds': {'rate': 0.03, 'default_prob': 0.05},
                'real_estate': {'returns': [0.12, 0.00, -0.05], 'probs': [0.6, 0.2, 0.2]},
                'stocks': {'returns': [0.20, -0.05], 'probs': [0.5, 0.5]}
            },
            'recession': {
                'cash': 0.02,
                'bonds': {'rate': 0.06, 'default_prob': 0.03},
                'real_estate': {'returns': [0.05, -0.05, -0.10], 'probs': [0.4, 0.4, 0.2]},
                'stocks': {'returns': [0.05, -0.10], 'probs': [0.3, 0.7]}
            }
        }

        # Initialize returns history
        self.returns_history = {asset: [] for asset in self.assets}

        # Generate returns for all 20 quarters
        for quarter in range(20):
            year = quarter // 4
            market_condition = self.market_conditions[year]
            for asset in self.assets:
                if asset == 'cash':
                    if callable(self.asset_distributions[market_condition][asset]):
                        return_val = self.asset_distributions[market_condition][asset]()
                    else:
                        return_val = self.asset_distributions[market_condition][asset]
                elif asset == 'bonds':
                    rate = self.asset_distributions[market_condition][asset]['rate']
                    default_prob = self.asset_distributions[market_condition][asset]['default_prob']
                    return_val = 0 if random.random() < default_prob else rate
                else:  # real_estate or stocks
                    dist = self.asset_distributions[market_condition][asset]
                    return_val = random.choices(dist['returns'], weights=dist['probs'])[0]
                self.returns_history[asset].append(return_val)

    def get_market_condition_for_quarter(self, quarter):

        if quarter < 0 or quarter >= 20:
            raise ValueError("Quarter must be between 0 and 19")
        year = quarter // 4
        return self.market_conditions[year]

    def get_historical_returns(self, up_to_quarter):

        if up_to_quarter < 0 or up_to_quarter > 20:
            raise ValueError("Quarter must be between 0 and 20")
        return {asset: self.returns_history[asset][:up_to_quarter] for asset in self.assets}

    def get_returns_for_quarter(self, quarter):

        if quarter < 0 or quarter >= 20:
            raise ValueError("Quarter must be between 0 and 19")
        return {asset: self.returns_history[asset][quarter] for asset in self.assets}

    def get_formatted_historical_returns(self, up_to_quarter):

        if up_to_quarter < 0 or up_to_quarter > 20:
            raise ValueError("Quarter must be between 0 and 20")

        if up_to_quarter == 0:
            return "No historical data available yet."

        # Get historical returns data
        historical_returns = self.get_historical_returns(up_to_quarter)

        # Format the output
        formatted_output = "Historical Asset Performance:\n\n"

        for quarter_idx in range(up_to_quarter):
            year = quarter_idx // 4 + 1  # Years are 1-based
            quarter_in_year = quarter_idx % 4 + 1  # Quarters are 1-based
            year_calendar = 1970 + year  # Starting from 1971

            # Get market condition for this quarter
            market_condition = self.get_market_condition_for_quarter(quarter_idx)

            # Add year and quarter header if this is the first quarter of the year
            if quarter_in_year == 1:
                formatted_output += f"Year {year} ({year_calendar}) - {market_condition.replace('_', ' ').title()} Market:\n"

            formatted_output += f"  Quarter {quarter_in_year}: "

            # Add returns for each asset
            for asset in self.assets:
                return_value = historical_returns[asset][quarter_idx]
                formatted_output += f"{asset.title()}: {return_value * 100:.1f}%, "

            # Remove trailing comma and space
            formatted_output = formatted_output.rstrip(", ")
            formatted_output += "\n"

            # Add a blank line between years
            if quarter_in_year == 4 and quarter_idx < up_to_quarter - 1:
                formatted_output += "\n"

        return formatted_output


# %% md
# # Shareholder
# %%
def _get_class_name(object_: object) -> str:
    return object_.__class__.__name__


# %%
class Shareholder:
    def __init__(self, players_profile, background_info, model):
        self.config = self.setup_agent_config(players_profile)
        self.clock = self.setup_clock()
        self.background_info = background_info
        self.memory = self.setup_memory()
        self.measurements = measurements_lib.Measurements()
        self.components = self.setup_components(self.measurements)
        self.agent = self.setup_agent(model)

    def setup_agent_config(self, profile):
        agent_config = formative_memories.AgentConfig(
            name=profile['profile']['name'],
            gender=profile['profile']['gender'],
            traits=profile['mbti'],
            extras={
                'leader': False,
                'profile': profile['profile'],
            }
        )
        return agent_config

    def setup_clock(self):
        """Initialize the simulation clock starting from 1971."""
        start_time = datetime.datetime(hour=9, year=1971, month=1, day=1)  # Set to 9AM, Jan 1, 1971
        major_time_setup = datetime.timedelta(minutes=5)  # 5-minute steps as specified
        minor_time_setup = datetime.timedelta(seconds=10)  # Keep minor step for detailed operations
        clock = game_clock.MultiIntervalClock(
            start=start_time,
            step_sizes=[major_time_setup, minor_time_setup])
        return clock

    def setup_memory(self):
        agent_memory = associative_memory.AssociativeMemory(
            sentence_embedder=embedder,
            clock=self.clock.now,
            importance=importance_model.importance,
        )
        # agent_memory.add(text=self.background_info.instructions_Shareholder,tags='instructions')
        agent_memory.add(text=self.background_info.role_shareholder, tags='role')
        agent_memory.add(text=self.background_info.annual_budget_meeting, tags='annual_budget_meeting')
        agent_memory.add(text=self.background_info.quarterly_investment_meetings, tags='quarterly_investment_meetings')
        agent_memory.add(text=self.background_info.annual_review_meeting, tags='annual_review_meeting')
        agent_memory.add(text=self.background_info.investment_detail, tags='investment_detail')
        profile_list = ['personality', 'advantages_and_disadvantages', 'hobby', 'growth_experience',
                        'family_relationship', 'working_conditions', 'social_relationship', 'emotional_state',
                        'living_conditions']
        for index in profile_list:
            if index == 'personality':
                text = f"{self.config.name}'s MBTI personality type is {self.config.traits}. "
                content = text + self.config.extras['profile'][index]
                index = index + '_MBTI'
                self.config.extras[index] = content
                agent_memory.add(text=content, tags=index, importance=1)
            else:
                content = self.config.extras['profile'][index]
                agent_memory.add(text=content, tags=index, importance=0.8)
        agent_memory = legacy_associative_memory.AssociativeMemoryBank(agent_memory)

        return agent_memory

    def setup_components(self, measurements: measurements_lib.Measurements | None = None):
        constant_personality = agent_components.constant.Constant(
            state=self.config.extras['personality_MBTI'],
            pre_act_key='Personality of MBTI'
        )
        simulation_background = agent_components.constant.Constant(
            state=self.background_info.instructions_Shareholder,
            pre_act_key='Simulation Background',
        )
        observation = agent_components.observation.Observation(
            clock_now=self.clock.now,
            timeframe=self.clock.get_step_size(),
            pre_act_key='\nObservation',
            logging_channel=measurements.get_channel('Observation').on_next,
        )
        observation_summary = agent_components.observation.ObservationSummary(
            model=model_low,
            clock_now=self.clock.now,
            timeframe_delta_from=datetime.timedelta(hours=4),
            timeframe_delta_until=datetime.timedelta(hours=0),
            pre_act_key='Summary of recent observations',
            logging_channel=measurements.get_channel('ObservationSummary').on_next,
        )
        SelfPerception = question_of_recent_memories.SelfPerception(
            model=model,
            pre_act_key='\nSelfPerception',
            clock_now=self.clock.now,
            components={'Personality of MBTI': constant_personality},
            logging_channel=measurements.get_channel('SelfPerception').on_next,
        )
        SituationPerception = question_of_recent_memories.SituationPerception(
            model=model,
            pre_act_key='\nSituationPerception',
            memory_tag='[situation reflection]',
            components={_get_class_name(observation): observation,
                        _get_class_name(observation_summary): observation_summary, },
            clock_now=self.clock.now,
            logging_channel=measurements.get_channel('SituationPerception').on_next,
        )
        PersonBySituation = question_of_recent_memories.PersonBySituation(
            model=model,
            pre_act_key='\nPersonBySituation',
            clock_now=self.clock.now,
            components={_get_class_name(SelfPerception): SelfPerception,
                        _get_class_name(SituationPerception): SituationPerception},
            logging_channel=measurements.get_channel('PersonBySituation').on_next,
        )
        all_components = {
            'Simulation Background': simulation_background,
            'Personality of MBTI': constant_personality,
            _get_class_name(observation): observation,
            _get_class_name(observation_summary): observation_summary,
            _get_class_name(SelfPerception): SelfPerception,
            _get_class_name(SituationPerception): SituationPerception,
            _get_class_name(PersonBySituation): PersonBySituation,
            agent_components.memory_component.DEFAULT_MEMORY_COMPONENT_NAME: agent_components.memory_component.MemoryComponent(
                self.memory),
        }
        return all_components

    def setup_agent(self, model):
        act_component = agent_components.concat_act_component.ConcatActComponent(
            model=model_high,
            clock=self.clock,
            component_order=self.components,
            logging_channel=self.measurements.get_channel('ActComponent').on_next,
        )
        agent = entity_agent_with_logging.EntityAgentWithLogging(
            agent_name=self.config.name,
            config=self.config,
            act_component=act_component,
            context_components=self.components,
            component_logging=self.measurements
        )
        return agent


# %% md
# # CEO
# %%
class CEO:
    def __init__(self, players_profile, background_info, model):
        self.config = self.setup_agent_config(players_profile)
        self.clock = self.setup_clock()
        self.background_info = background_info
        self.memory = self.setup_memory()
        self.measurements = measurements_lib.Measurements()
        self.components = self.setup_components(self.measurements)
        self.agent = self.setup_agent(model)

    def setup_agent_config(self, profile):
        agent_config = formative_memories.AgentConfig(
            name=profile['profile']['name'],
            gender=profile['profile']['gender'],
            traits=profile['mbti'],
            extras={
                'leader': True,
                'profile': profile['profile'],
            }
        )
        return agent_config

    def setup_clock(self):
        """Initialize the simulation clock starting from 1971."""
        start_time = datetime.datetime(hour=9, year=1971, month=1, day=1)  # Set to 9AM, Jan 1, 1971
        major_time_setup = datetime.timedelta(minutes=5)  # 5-minute steps as specified
        minor_time_setup = datetime.timedelta(seconds=10)  # Keep minor step for detailed operations
        clock = game_clock.MultiIntervalClock(
            start=start_time,
            step_sizes=[major_time_setup, minor_time_setup])
        return clock

    def setup_memory(self):
        agent_memory = associative_memory.AssociativeMemory(
            sentence_embedder=embedder,
            clock=self.clock.now,
            importance=importance_model.importance,
        )
        # agent_memory.add(text=self.background_info.instructions_Shareholder,tags='instructions')
        agent_memory.add(text=self.background_info.role_ceo, tags='role')
        agent_memory.add(text=self.background_info.annual_budget_meeting, tags='annual_budget_meeting')
        agent_memory.add(text=self.background_info.quarterly_investment_meetings, tags='quarterly_investment_meetings')
        agent_memory.add(text=self.background_info.annual_review_meeting, tags='annual_review_meeting')
        agent_memory.add(text=self.background_info.investment_detail, tags='investment_detail')
        profile_list = ['personality', 'advantages_and_disadvantages', 'hobby', 'growth_experience',
                        'family_relationship', 'working_conditions', 'social_relationship', 'emotional_state',
                        'living_conditions']
        for index in profile_list:
            if index == 'personality':
                text = f"{self.config.name}'s MBTI personality type is {self.config.traits}. "
                content = text + self.config.extras['profile'][index]
                index = index + '_MBTI'
                self.config.extras[index] = content
                agent_memory.add(text=content, tags=index, importance=1)
            else:
                content = self.config.extras['profile'][index]
                agent_memory.add(text=content, tags=index, importance=0.8)
        agent_memory = legacy_associative_memory.AssociativeMemoryBank(agent_memory)

        return agent_memory

    def setup_components(self, measurements: measurements_lib.Measurements | None = None):
        constant_personality = agent_components.constant.Constant(
            state=self.config.extras['personality_MBTI'],
            pre_act_key='Personality of MBTI'
        )
        simulation_background = agent_components.constant.Constant(
            state=self.background_info.instructions_Shareholder,
            pre_act_key='Simulation Background',
        )
        observation = agent_components.observation.Observation(
            clock_now=self.clock.now,
            timeframe=self.clock.get_step_size(),
            pre_act_key='\nObservation',
            logging_channel=measurements.get_channel('Observation').on_next,
        )
        observation_summary = agent_components.observation.ObservationSummary(
            model=model_low,
            clock_now=self.clock.now,
            timeframe_delta_from=datetime.timedelta(hours=4),
            timeframe_delta_until=datetime.timedelta(hours=0),
            pre_act_key='Summary of recent observations',
            logging_channel=measurements.get_channel('ObservationSummary').on_next,
        )
        SelfPerception = question_of_recent_memories.SelfPerception(
            model=model,
            pre_act_key='\nSelfPerception',
            clock_now=self.clock.now,
            components={'Personality of MBTI': constant_personality},
            logging_channel=measurements.get_channel('SelfPerception').on_next,
        )
        SituationPerception = question_of_recent_memories.SituationPerception(
            model=model,
            pre_act_key='\nSituationPerception',
            memory_tag='[situation reflection]',
            components={_get_class_name(observation): observation,
                        _get_class_name(observation_summary): observation_summary, },
            clock_now=self.clock.now,
            logging_channel=measurements.get_channel('SituationPerception').on_next,
        )
        PersonBySituation = question_of_recent_memories.PersonBySituation(
            model=model,
            pre_act_key='\nPersonBySituation',
            clock_now=self.clock.now,
            components={_get_class_name(SelfPerception): SelfPerception,
                        _get_class_name(SituationPerception): SituationPerception},
            logging_channel=measurements.get_channel('PersonBySituation').on_next,
        )
        all_components = {
            'Simulation Background': simulation_background,
            'Personality of MBTI': constant_personality,
            _get_class_name(observation): observation,
            _get_class_name(observation_summary): observation_summary,
            _get_class_name(SelfPerception): SelfPerception,
            _get_class_name(SituationPerception): SituationPerception,
            _get_class_name(PersonBySituation): PersonBySituation,
            agent_components.memory_component.DEFAULT_MEMORY_COMPONENT_NAME: agent_components.memory_component.MemoryComponent(
                self.memory),
        }
        return all_components

    def setup_agent(self, model):
        act_component = agent_components.concat_act_component.ConcatActComponent(
            model=model_high,
            clock=self.clock,
            component_order=self.components,
            logging_channel=self.measurements.get_channel('ActComponent').on_next,
        )
        agent = entity_agent_with_logging.EntityAgentWithLogging(
            agent_name=self.config.name,
            config=self.config,
            act_component=act_component,
            context_components=self.components,
            component_logging=self.measurements
        )
        return agent


# %% md
# # Company
# %%

class Company:
    def __init__(self, shareholders, ceo, market, prompts, initial_assets=100000, model=None,simulation=None):
        self.shareholders = shareholders
        self.ceo = ceo
        self.market = market
        self.assets = initial_assets
        self.assets_logs = [initial_assets]  # Track asset changes by quarter
        self.investment_decisions = []  # Track investment decisions
        # Record initial company assets
        if hasattr(self, 'decision_tracker'):
            self.decision_tracker.record_company_assets(initial_assets, is_final=False)
        self.meeting_logs = []  # Track meeting discussions and decisions
        self.shareholder_ratings = []  # Track shareholder ratings of CEO and company
        self.budget = []  # Annual investment budget plans
        self.prompts = prompts
        self.model = model
        self.simulation = simulation  # Store reference to simulation

        # Get the embedder from the CEO object for document retrieval
        self.embedder = Model_And_Embedder.embedder  # Use the global embedder

        # NEW: Initialize DecisionTracker
        self.decision_tracker = DecisionTracker(ceo.config.name, ceo.config.traits, shareholders)

        # Initialize meeting manager with document handling
        self.meeting_manager = MeetingManager(self, model)

        # Pass market condition intros to meeting manager if available
        if hasattr(self, 'background_info'):
            self.meeting_manager.market_condition_intros = self.background_info.market_condition_intro

    def run_annual_cycle(self, year):
        year_results = {
            "year": year,
            "market_condition": self.market.market_conditions[year - 1],
            "starting_assets": self.assets,
            "quarterly_results": []
        }

        # Add year transition document entry
        year_transition_text = f"# Year {year} ({1970 + year})\n\nThe company is now entering Year {year} with ${self.assets:.2f} in assets.\n"
        year_transition_text += f"Market condition: {self.market.market_conditions[year - 1]}\n\n"

        # If we have a document manager, create a year document
        if hasattr(self.meeting_manager, 'document_manager'):
            # Add it to all master documents
            self.meeting_manager.document_manager.add_to_all_documents(year_transition_text)

        # 1. Annual Budget Meeting
        print(f"\n=== Year {year} Annual Budget Meeting ===")
        budget_option, budget_log = self._conduct_annual_budget_meeting(year)
        self.meeting_logs.append(budget_log)
        year_results["budget_option"] = budget_option
        # time.sleep(60)
        # Record annual budget decision
        if hasattr(self, 'decision_tracker'):
            self.decision_tracker.record_company_decision(
                meeting_id=f"annual_budget_year{year}",
                final_choice=budget_option,
                voting_details=self.meeting_manager.voting_manager.votes.get(f"annual_budget_year{year}",
                                                                             {}) if hasattr(self.meeting_manager,
                                                                                            'voting_manager') else {}
            )
        # 2. Quarterly Investment Meetings and Investment Processing
        for quarter in range(1, 5):
            print(f"\n=== Year {year}, Quarter {quarter} Investment Meeting ===")
            asset_option, investment_log = self._conduct_quarterly_investment_meeting(year, quarter)
            self.meeting_logs.append(investment_log)

            # Record quarterly investment decision
            if hasattr(self, 'decision_tracker'):
                self.decision_tracker.record_company_decision(
                    meeting_id=f"quarterly_investment_year{year}_q{quarter}",
                    final_choice=asset_option,
                    voting_details=self.meeting_manager.voting_manager.votes.get(
                        f"quarterly_investment_year{year}_q{quarter}", {}) if hasattr(self.meeting_manager,
                                                                                      'voting_manager') else {}
                )

            # Process investment returns
            quarterly_result = self._process_investment_returns(year, quarter, asset_option)
            year_results["quarterly_results"].append(quarterly_result)
            # Record asset changes
            if hasattr(self, 'decision_tracker'):
                self.decision_tracker.record_company_assets(self.assets, is_final=False)
            # time.sleep(300)

        # 3. Annual Review Meeting
        print(f"\n=== Year {year} Annual Review Meeting ===")
        ratings, review_log = self._conduct_annual_review_meeting(year)
        self.meeting_logs.append(review_log)

        year_results["ceo_rating"] = ratings["ceo_rating"]
        year_results["company_rating"] = ratings["company_rating"]
        year_results["ending_assets"] = self.assets
        year_results["growth_percentage"] = ((self.assets / year_results["starting_assets"]) - 1) * 100

        self.shareholder_ratings.append(ratings)

        # Record company review results
        if hasattr(self, 'decision_tracker'):
            # Extract detailed ratings from the review results
            detailed_ratings = ratings.get("detailed_ratings", {})

            company_ratings = {
                "average_score": ratings["company_rating"],
                "detailed_scores": detailed_ratings.get("company", {}),
                "individual_ratings": {}
            }

            ceo_ratings = {
                "average_score": ratings["ceo_rating"],
                "detailed_scores": detailed_ratings.get("ceo", {}),
                "individual_ratings": {}
            }

            # Extract individual ratings if available
            if "company" in detailed_ratings:
                for dimension, data in detailed_ratings["company"].items():
                    for shareholder, rating_info in data.items():
                        if shareholder not in company_ratings["individual_ratings"]:
                            company_ratings["individual_ratings"][shareholder] = {}
                        company_ratings["individual_ratings"][shareholder][dimension] = {
                            "score": rating_info.get("score", 0),
                            "reason": rating_info.get("explanation", "")
                        }

            if "ceo" in detailed_ratings:
                for dimension, data in detailed_ratings["ceo"].items():
                    for shareholder, rating_info in data.items():
                        if shareholder not in ceo_ratings["individual_ratings"]:
                            ceo_ratings["individual_ratings"][shareholder] = {}
                        ceo_ratings["individual_ratings"][shareholder][dimension] = {
                            "score": rating_info.get("score", 0),
                            "reason": rating_info.get("explanation", "")
                        }

            self.decision_tracker.record_company_review(
                meeting_id=f"annual_review_year{year}",
                company_ratings=company_ratings,
                ceo_ratings=ceo_ratings
            )
        # Add year summary to document
        year_summary_text = f"# Year {year} Summary\n\n"
        year_summary_text += f"Starting assets: ${year_results['starting_assets']:.2f}\n"
        year_summary_text += f"Ending assets: ${year_results['ending_assets']:.2f}\n"
        year_summary_text += f"Growth: {year_results['growth_percentage']:.2f}%\n\n"
        year_summary_text += f"CEO rating: {year_results['ceo_rating']:.1f}/10\n"
        year_summary_text += f"Company rating: {year_results['company_rating']:.1f}/10\n\n"

        # Add quarterly results
        year_summary_text += "## Quarterly Results\n\n"
        for q in year_results['quarterly_results']:
            year_summary_text += f"Q{q['quarter']}: Invested ${q['budget']:.2f} in {q['asset']}, "
            year_summary_text += f"Return: {q['return_rate'] * 100:.1f}% (${q['return_amount']:.2f})\n"

        # If we have a document manager, add the summary
        if hasattr(self.meeting_manager, 'document_manager'):
            self.meeting_manager.document_manager.add_to_all_documents(year_summary_text)
        # Record final assets for the year
        if hasattr(self, 'decision_tracker'):
            # Always record as final when it's the last year of simulation, not hardcoded to year 5
            self.decision_tracker.record_company_assets(self.assets, is_final=True)
        return year_results

    def _conduct_annual_budget_meeting(self, year):
        # Prepare meeting inputs
        meeting_id = f"annual_budget_year{year}"
        market_condition = self.market.get_market_condition_for_quarter((year - 1) * 4)

        # Notify all participants about the meeting
        self._notify_meeting_start("annual_budget_meeting", year)

        # Add transition text to document if we have a document manager
        if hasattr(self.meeting_manager, 'document_manager'):
            transition_text = f"## Annual Budget Meeting Preparation\n\nCEO {self.ceo.config.name} is preparing for the annual budget meeting for Year {year}.\n\n"
            self.meeting_manager.document_manager.add_to_agent_only(self.ceo.config.name, transition_text,
                                                                    tags=["meeting_preparation"])

        # Define budget options
        budget_options = {
            "Option A": "50% of total funds, evenly distributed (12.5% per quarter)",
            "Option B": "50% of total funds, increasing allocation (5%, 10%, 15%, 20%)",
            "Option C": "100% of total funds, evenly distributed (25% per quarter)",
            "Option D": "100% of total funds, increasing allocation (10%, 20%, 30%, 40%)"
        }

        # Get historical returns
        historical_returns = None
        if year > 1:
            historical_returns = self.market.get_historical_returns((year - 1) * 4)

        # Run the public discussion
        selected_option, discussion_log = self.meeting_manager.run_public_discussion(
            meeting_id,
            "annual_budget",
            budget_options,
            market_condition,
            historical_returns
        )

        # Process the budget decision
        self._process_budget_decision(selected_option, year)

        # Notify meeting end
        self._notify_meeting_end("annual_budget_meeting", year, selected_option)

        return selected_option, discussion_log

    def _conduct_quarterly_investment_meeting(self, year, quarter):
        # Prepare meeting inputs
        meeting_id = f"quarterly_investment_year{year}_q{quarter}"
        market_condition = self.market.get_market_condition_for_quarter((year - 1) * 4 + quarter - 1)

        # Notify all participants about the meeting
        self._notify_meeting_start("quarterly_investment_meetings", year, quarter)

        # Add transition text to document if we have a document manager
        if hasattr(self.meeting_manager, 'document_manager'):
            transition_text = f"## Quarterly Investment Meeting Preparation\n\nCEO {self.ceo.config.name} is preparing for the quarterly investment meeting for Year {year}, Quarter {quarter}.\n\n"
            self.meeting_manager.document_manager.add_to_agent_only(self.ceo.config.name, transition_text,
                                                                    tags=["meeting_preparation"])

        # Define asset options
        asset_options = {
            "Cash": "Low risk, low return",
            "Bonds": "Low-medium risk, low-medium return",
            "Real Estate": "Medium risk, medium return",
            "Stocks": "High risk, high return"
        }

        # Calculate quarterly budget
        quarterly_budget = self._calculate_quarterly_budget(year, quarter)

        # Get historical returns
        up_to_quarter = (year - 1) * 4 + quarter - 1
        historical_returns = self.market.get_historical_returns(up_to_quarter)

        # Run the public discussion
        selected_asset, discussion_log = self.meeting_manager.run_public_discussion(
            meeting_id,
            "quarterly_investment",
            asset_options,
            market_condition,
            historical_returns,
            quarterly_budget
        )

        # Notify meeting end
        self._notify_meeting_end("quarterly_investment_meetings", year, selected_asset, quarter)

        return selected_asset, discussion_log

    def _conduct_annual_review_meeting(self, year):
        # Prepare meeting inputs
        meeting_id = f"annual_review_year{year}"

        # Notify all participants about the meeting
        self._notify_meeting_start("annual_review_meeting", year)

        # Add transition text to document if we have a document manager
        if hasattr(self.meeting_manager, 'document_manager'):
            transition_text = f"## Annual Review Meeting Preparation\n\nCEO {self.ceo.config.name} is preparing for the annual review meeting for Year {year}.\n\n"
            self.meeting_manager.document_manager.add_to_agent_only(self.ceo.config.name, transition_text,
                                                                    tags=["meeting_preparation"])

        # Gather yearly performance data
        yearly_performance = self._gather_yearly_performance(year)

        # Run the written submission process
        ratings, meeting_log = self.meeting_manager.run_written_submission(
            meeting_id,
            "annual_review",
            yearly_performance
        )

        # Notify meeting end
        self._notify_meeting_end("annual_review_meeting", year, ratings)

        return ratings, meeting_log

    def _process_budget_decision(self, selected_option, year):
        # Extract option letter
        option_letter = selected_option.split()[-1] if "Option " in selected_option else selected_option

        budget_mappings = {
            "Option A": {"distribution": "even", "percentage": 0.5},  # 50% evenly distributed
            "A": {"distribution": "even", "percentage": 0.5},
            "Option B": {"distribution": "increasing", "percentage": 0.5},  # 50% increasingly distributed
            "B": {"distribution": "increasing", "percentage": 0.5},
            "Option C": {"distribution": "even", "percentage": 1.0},  # 100% evenly distributed
            "C": {"distribution": "even", "percentage": 1.0},
            "Option D": {"distribution": "increasing", "percentage": 1.0},  # 100% increasingly distributed
            "D": {"distribution": "increasing", "percentage": 1.0}
        }

        selected_budget = budget_mappings.get(option_letter, {"distribution": "even", "percentage": 0.5})
        self.budget.append(selected_budget)

        # Document the budget decision if we have a document manager
        if hasattr(self.meeting_manager, 'document_manager'):
            decision_text = f"## Budget Decision for Year {year}\n\n"

            if selected_budget["distribution"] == "even":
                if selected_budget["percentage"] == 0.5:
                    quarterly_allocations = "12.5% per quarter"
                    detailed_allocation = "12.5%, 12.5%, 12.5%, 12.5%"
                else:
                    quarterly_allocations = "25% per quarter"
                    detailed_allocation = "25%, 25%, 25%, 25%"
            else:  # increasing
                if selected_budget["percentage"] == 0.5:
                    quarterly_allocations = "5%, 10%, 15%, 20%"
                    detailed_allocation = quarterly_allocations
                else:
                    quarterly_allocations = "10%, 20%, 30%, 40%"
                    detailed_allocation = quarterly_allocations

            decision_text += f"The budget decision for year {year} was {selected_option}:\n"
            decision_text += f"- {selected_budget['percentage'] * 100}% of total funds will be allocated\n"
            decision_text += f"- Distribution method: {selected_budget['distribution']}\n"
            decision_text += f"- Quarterly allocation: {detailed_allocation}\n\n"

            self.meeting_manager.document_manager.add_to_all_documents(decision_text, tags=["budget_decision"])

        # Announce the budget decision
        announcement = f"Budget decision for year {year}: {selected_budget['percentage'] * 100}% of total funds will be allocated with {quarterly_allocations}."
        self.everyone_observe(announcement)

    def _calculate_quarterly_budget(self, year, quarter):
        if year <= len(self.budget):
            budget_info = self.budget[year - 1]
            total_percentage = budget_info["percentage"]

            if budget_info["distribution"] == "even":
                return self.assets * total_percentage / 4
            else:  # increasing
                if total_percentage == 0.5:
                    quarterly_percentages = [0.05, 0.10, 0.15, 0.20]
                else:
                    quarterly_percentages = [0.10, 0.20, 0.30, 0.40]

                return self.assets * quarterly_percentages[quarter - 1]
        else:
            # Default if no budget decision yet
            return self.assets * 0.125  # 12.5% per quarter

    def _process_investment_returns(self, year, quarter, selected_asset):
        # Convert asset name to internal format if needed
        asset_mapping = {
            "Cash": "cash",
            "Bonds": "bonds",
            "Real Estate": "real_estate",
            "Stocks": "stocks"
        }

        internal_asset = asset_mapping.get(selected_asset, selected_asset.lower())

        # Calculate the quarterly budget
        quarterly_budget = self._calculate_quarterly_budget(year, quarter)

        # Record the investment decision
        investment_record = {
            "year": year,
            "quarter": quarter,
            "asset": internal_asset,
            "budget": quarterly_budget
        }
        self.investment_decisions.append(investment_record)

        # Process returns at the end of the quarter
        quarter_idx = (year - 1) * 4 + quarter - 1
        returns = self.market.get_returns_for_quarter(quarter_idx)[internal_asset]

        # Update company assets
        investment_return = quarterly_budget * returns
        self.assets += investment_return
        self.assets_logs.append(self.assets)

        # Update investment record with returns
        investment_record["return_rate"] = returns
        investment_record["return_amount"] = investment_return
        # Record quarterly investment in company assets and decision tracking
        if hasattr(self, 'decision_tracker'):
            self.decision_tracker.record_quarterly_investment(
                meeting_id=f"quarterly_investment_year{year}_q{quarter}",
                invested_amount=quarterly_budget,
                return_amount=investment_return,
                return_rate=returns
            )
        # Document the investment results if we have a document manager
        if hasattr(self.meeting_manager, 'document_manager'):
            results_text = f"## Investment Results - Year {year}, Quarter {quarter}\n\n"
            results_text += f"Asset: {selected_asset}\n"
            results_text += f"Budget: ${quarterly_budget:.2f}\n"
            results_text += f"Return rate: {returns * 100:.1f}%\n"
            results_text += f"Return amount: ${investment_return:.2f}\n"
            results_text += f"Company assets after investment: ${self.assets:.2f}\n\n"

            self.meeting_manager.document_manager.add_to_all_documents(results_text, tags=["investment_results"])

        # Announce the investment results
        announcement = f"Investment results for year {year}, quarter {quarter}: The {quarterly_budget:.2f} invested in {selected_asset} yielded a return of {returns * 100:.1f}%, resulting in a gain/loss of {investment_return:.2f}. Company assets are now {self.assets:.2f}."
        self.everyone_observe(announcement)

        return {
            "quarter": quarter,
            "asset": selected_asset,
            "budget": quarterly_budget,
            "return_rate": returns,
            "return_amount": investment_return,
            "assets_after": self.assets
        }

    def _gather_yearly_performance(self, year):
        start_idx = (year - 1) * 4
        end_idx = year * 4 - 1

        yearly_performance = {
            "year": year,
            "market_condition": self.market.market_conditions[year - 1],
            "starting_assets": self.assets_logs[start_idx] if start_idx < len(self.assets_logs) else 100000,
            "ending_assets": self.assets_logs[end_idx] if end_idx < len(self.assets_logs) else self.assets,
            "quarterly_investments": [],
            "total_return": 0
        }

        # Calculate total return
        yearly_performance["total_return"] = (yearly_performance["ending_assets"] - yearly_performance[
            "starting_assets"]) / yearly_performance["starting_assets"] * 100

        # Gather quarterly investments
        for q in range(1, 5):
            quarter_idx = (year - 1) * 4 + q - 1
            investment_info = next(
                (inv for inv in self.investment_decisions if inv["year"] == year and inv["quarter"] == q), None)

            if investment_info:
                yearly_performance["quarterly_investments"].append(investment_info)

        return yearly_performance

    def _notify_meeting_start(self, meeting_type, year, quarter=None):
        input_variables = {
            "agent_name": None,  # Will be replaced in the notification function
            "ceo_name": self.ceo.config.name,
            "year": year
        }

        if meeting_type == "annual_budget_meeting":
            template = "This year is the {year}th year since the company was founded. Currently, {agent_name}, as one of the company's shareholders, is attending the annual budget meeting at the beginning of this year, which is being chaired by the company's CEO, {ceo_name}."
            document_text = f"## Annual Budget Meeting - Year {year}\n\nThe annual budget meeting for Year {year} has begun, chaired by CEO {self.ceo.config.name}.\n\n"
        elif meeting_type == "annual_review_meeting":
            template = "This year is the {year}th year since the company was founded. Currently, {agent_name}, as one of the company's shareholders, is attending the annual review meeting at the end of this year, which is being chaired by the company's CEO, {ceo_name}."
            document_text = f"## Annual Review Meeting - Year {year}\n\nThe annual review meeting for Year {year} has begun, chaired by CEO {self.ceo.config.name}.\n\n"
        elif meeting_type == "quarterly_investment_meetings":
            if quarter is None:
                raise ValueError("Quarter must be provided for quarterly investment meetings")
            input_variables["quarter"] = quarter
            template = "This year is the {year}th year since the company was founded. Currently, {agent_name}, as one of the company's shareholders, is attending the {quarter}th quarterly investment meeting of this year, which is being chaired by the company's CEO, {ceo_name}."
            document_text = f"## Quarterly Investment Meeting - Year {year}, Quarter {quarter}\n\nThe quarterly investment meeting for Year {year}, Quarter {quarter} has begun, chaired by CEO {self.ceo.config.name}.\n\n"
        else:
            raise ValueError(f"Unknown meeting type: {meeting_type}")

        # Initialize documents for this meeting
        if hasattr(self.meeting_manager, 'document_manager'):
            meeting_id = f"{meeting_type}_year{year}" + (f"_q{quarter}" if quarter else "")
            self.meeting_manager.document_manager.initialize_meeting(meeting_id, meeting_type, year, quarter)
            self.meeting_manager.document_manager.add_to_all_documents(document_text, tags=["meeting_start"])

        # Notify all participants using the traditional observe method
        self.everyone_observe(template, input_variables)

    def _notify_meeting_end(self, meeting_type, year, result, quarter=None):
        message = f"The {meeting_type.replace('_', ' ')} for year {year}"
        if quarter:
            message += f", quarter {quarter}"
        message += " has concluded with the following result: " + escape_curly_braces(str(result)) + "."

        # Document the meeting end if we have a document manager
        if hasattr(self.meeting_manager, 'document_manager'):
            end_text = f"## Meeting Conclusion\n\n{message}\n\n"
            self.meeting_manager.document_manager.add_to_all_documents(end_text, tags=["meeting_end"])

            # Finalize the meeting documents
            if hasattr(self.meeting_manager.document_manager, 'finalize_meeting'):
                self.meeting_manager.document_manager.finalize_meeting()

        # Notify all participants
        self.everyone_observe(message)

    def everyone_observe(self, observe_text, input_variables=None):
        if input_variables is None:
            input_variables = {}

        def _pool_observe(agent, observe_text, input_variables):
            if 'agent_name' in input_variables:
                input_variables['agent_name'] = agent.config.name
            formatted_text = observe_text.format(**input_variables)

            agent.agent.observe(formatted_text)
            return_dict = {
                'agent_name': agent.config.name,
                'observe_text': formatted_text
            }
            return return_dict

        loop_iter = len(self.shareholders.values()) + 1  # +1 for CEO
        all_agents = list(self.shareholders.values()) + [self.ceo]

        with concurrent.futures.ThreadPoolExecutor(max_workers=loop_iter) as pool:
            for result in pool.map(_pool_observe,
                                   all_agents,
                                   [observe_text] * loop_iter,
                                   [input_variables] * loop_iter):
                if self.prompts.get('verbose', False):
                    print(f"{result['agent_name']} observed: {result['observe_text']}")

    def shareholders_observe(self, observe_text, input_variables=None):
        if input_variables is None:
            input_variables = {}

        def _pool_observe(agent, observe_text, input_variables):
            formatted_text = observe_text
            if 'agent_name' in input_variables:
                formatted_text = observe_text.format(agent_name=agent.config.name, **input_variables)
            elif input_variables:
                formatted_text = observe_text.format(**input_variables)

            agent.agent.observe(formatted_text)
            return_dict = {
                'agent_name': agent.config.name,
                'observe_text': formatted_text
            }
            return return_dict

        loop_iter = len(self.shareholders.values())

        with concurrent.futures.ThreadPoolExecutor(max_workers=loop_iter) as pool:
            for result in pool.map(_pool_observe,
                                   self.shareholders.values(),
                                   [observe_text] * loop_iter,
                                   [input_variables] * loop_iter):
                if self.prompts.get('verbose', False):
                    print(f"{result['agent_name']} observed: {result['observe_text']}")

    def get_formatted_investment_history(self, up_to_year, up_to_quarter):
        if not self.investment_decisions:
            return "No previous investment decisions have been made yet."

        # Filter decisions up to the specified year and quarter
        filtered_decisions = []
        for decision in self.investment_decisions:
            if (decision["year"] < up_to_year) or \
                    (decision["year"] == up_to_year and decision["quarter"] < up_to_quarter):
                filtered_decisions.append(decision)

        if not filtered_decisions:
            return "No previous investment decisions have been made yet."

        # Format the output
        formatted_output = "Previous Investment Decisions:\n\n"

        # Group by year
        years = sorted(set(decision["year"] for decision in filtered_decisions))
        for year in years:
            year_calendar = 1970 + year  # Starting from 1971
            year_decisions = [d for d in filtered_decisions if d["year"] == year]

            # Get market condition for this year
            market_condition = self.market.market_conditions[year - 1]

            formatted_output += f"Year {year} ({year_calendar}) - {market_condition.replace('_', ' ').title()} Market:\n"

            # Sort by quarter
            year_decisions.sort(key=lambda d: d["quarter"])
            for decision in year_decisions:
                quarter = decision["quarter"]
                asset = decision["asset"]
                budget = decision["budget"]
                return_rate = decision.get("return_rate", 0)
                return_amount = decision.get("return_amount", 0)

                formatted_output += f"  Quarter {quarter}: "
                formatted_output += f"Invested ${budget:.2f} in {asset.title()}, "
                formatted_output += f"Return: {return_rate * 100:.1f}% (${return_amount:.2f})\n"

            # Add a blank line between years
            if year < years[-1]:
                formatted_output += "\n"

        return formatted_output

    def get_formatted_budget_history(self, up_to_year):
        if not self.budget or up_to_year <= 1:
            return "No previous budget decisions have been made yet."

        # Format the output
        formatted_output = "Previous Budget Decisions and Performance:\n\n"

        # Process each previous year
        for year in range(1, up_to_year):
            year_calendar = 1970 + year  # Starting from 1971
            market_condition = self.market.market_conditions[year - 1]

            # Get the budget decision for this year
            budget_info = self.budget[year - 1]
            total_percentage = budget_info["percentage"] * 100  # Convert to percentage
            distribution = budget_info["distribution"]

            # Calculate what the quarterly allocations were
            if distribution == "even":
                if total_percentage == 50:
                    allocation_str = "evenly distributed (12.5% per quarter)"
                    quarters_str = "12.5%, 12.5%, 12.5%, 12.5%"
                else:  # 100%
                    allocation_str = "evenly distributed (25% per quarter)"
                    quarters_str = "25%, 25%, 25%, 25%"
            else:  # increasing
                if total_percentage == 50:
                    allocation_str = "increasing allocation (5%, 10%, 15%, 20%)"
                    quarters_str = "5%, 10%, 15%, 20%"
                else:  # 100%
                    allocation_str = "increasing allocation (10%, 20%, 30%, 40%)"
                    quarters_str = "10%, 20%, 30%, 40%"

            # Get all investment decisions for this year
            year_decisions = [d for d in self.investment_decisions if d["year"] == year]

            # Calculate total returns for the year
            total_investment = sum(d["budget"] for d in year_decisions)
            total_returns = sum(d.get("return_amount", 0) for d in year_decisions)
            if total_investment > 0:
                return_percentage = (total_returns / total_investment) * 100
            else:
                return_percentage = 0

            # Find starting and ending assets for the year
            start_idx = (year - 1) * 4
            end_idx = year * 4
            if start_idx < len(self.assets_logs) and end_idx < len(self.assets_logs):
                starting_assets = self.assets_logs[start_idx]
                ending_assets = self.assets_logs[end_idx]
                growth_percentage = ((ending_assets / starting_assets) - 1) * 100
            else:
                starting_assets = 0
                ending_assets = 0
                growth_percentage = 0

            # Format the year's information
            formatted_output += f"Year {year} ({year_calendar}) - {market_condition.replace('_', ' ').title()} Market:\n"
            formatted_output += f"  Budget: {total_percentage:.0f}% of total funds, {allocation_str}\n"
            formatted_output += f"  Quarterly allocation: {quarters_str}\n"
            formatted_output += f"  Starting assets: ${starting_assets:.2f}, Ending assets: ${ending_assets:.2f}\n"
            formatted_output += f"  Overall growth: {growth_percentage:.2f}%, Return on investment: {return_percentage:.2f}%\n"

            # Add quarterly investment details
            formatted_output += "  Quarterly investments:\n"
            for quarter in range(1, 5):
                quarter_decision = next((d for d in year_decisions if d["quarter"] == quarter), None)
                if quarter_decision:
                    asset = quarter_decision["asset"]
                    budget = quarter_decision["budget"]
                    return_rate = quarter_decision.get("return_rate", 0)
                    return_amount = quarter_decision.get("return_amount", 0)

                    formatted_output += f"    Q{quarter}: ${budget:.2f} in {asset.title()}, "
                    formatted_output += f"Return: {return_rate * 100:.1f}% (${return_amount:.2f})\n"
                else:
                    formatted_output += f"    Q{quarter}: No data available\n"

            # Add a blank line between years
            if year < up_to_year - 1:
                formatted_output += "\n"

        return formatted_output

    def get_formatted_annual_performance_history(self, current_year):
        # Format the output with a clear separation between current and historical
        formatted_output = "Company Performance Review:\n\n"

        # PART 1: CURRENT YEAR PERFORMANCE
        market_condition = self.market.market_conditions[current_year - 1]
        year_calendar = 1970 + current_year

        # Calculate current year metrics
        start_idx = (current_year - 1) * 4
        end_idx = current_year * 4

        if start_idx < len(self.assets_logs) and end_idx < len(self.assets_logs):
            starting_assets = self.assets_logs[start_idx]
            ending_assets = self.assets_logs[end_idx]
            growth_percentage = ((ending_assets / starting_assets) - 1) * 100
        else:
            # Fallback if asset logs aren't complete
            starting_assets = self.assets_logs[start_idx] if start_idx < len(self.assets_logs) else 0
            ending_assets = self.assets
            growth_percentage = ((ending_assets / starting_assets) - 1) * 100 if starting_assets > 0 else 0

        # Get current year's budget decision
        budget_info = self.budget[current_year - 1] if current_year <= len(self.budget) else None
        if budget_info:
            total_percentage = budget_info["percentage"] * 100
            distribution = budget_info["distribution"]

            if distribution == "even":
                if total_percentage == 50:
                    budget_str = "50% of funds, evenly distributed (12.5% per quarter)"
                else:  # 100%
                    budget_str = "100% of funds, evenly distributed (25% per quarter)"
            else:  # increasing
                if total_percentage == 50:
                    budget_str = "50% of funds, increasing allocation (5%, 10%, 15%, 20%)"
                else:  # 100%
                    budget_str = "100% of funds, increasing allocation (10%, 20%, 30%, 40%)"
        else:
            budget_str = "No budget data available"

        # Get current year's investment decisions
        year_decisions = [d for d in self.investment_decisions if d["year"] == current_year]

        # Calculate total returns for the year
        total_investment = sum(d["budget"] for d in year_decisions)
        total_returns = sum(d.get("return_amount", 0) for d in year_decisions)

        if total_investment > 0:
            return_percentage = (total_returns / total_investment) * 100
        else:
            return_percentage = 0

        # SECTION 1: CURRENT YEAR SUMMARY
        formatted_output += f"## CURRENT YEAR {current_year} ({year_calendar}) - {market_condition.replace('_', ' ').title()} Market\n\n"
        formatted_output += f"Budget Strategy: {budget_str}\n"
        formatted_output += f"Starting Assets: ${starting_assets:.2f}\n"
        formatted_output += f"Ending Assets: ${ending_assets:.2f}\n"
        formatted_output += f"Asset Growth: {growth_percentage:.2f}%\n"
        formatted_output += f"Return on Investment: {return_percentage:.2f}%\n\n"

        # SECTION 2: CURRENT YEAR QUARTERLY BREAKDOWN
        formatted_output += "Quarterly Investment Decisions:\n"

        for quarter in range(1, 5):
            quarter_decision = next((d for d in year_decisions if d["quarter"] == quarter), None)
            if quarter_decision:
                asset = quarter_decision["asset"]
                budget = quarter_decision["budget"]
                return_rate = quarter_decision.get("return_rate", 0)
                return_amount = quarter_decision.get("return_amount", 0)

                formatted_output += f"  Q{quarter}: ${budget:.2f} in {asset.title()}, "
                formatted_output += f"Return: {return_rate * 100:.1f}% (${return_amount:.2f})\n"
            else:
                formatted_output += f"  Q{quarter}: No data available\n"

        # SECTION 3: MARKET CONDITION COMPARISON
        formatted_output += "\nPerformance in Current Market Context:\n"
        formatted_output += f"  The company operated in a {market_condition.replace('_', ' ').title()} Market this year.\n"

        # Compare to the theoretical maximums for the year
        theoretical_max_return = 0
        for quarter in range(1, 5):
            quarter_idx = (current_year - 1) * 4 + quarter - 1
            if quarter_idx < len(self.market.returns_history["cash"]):
                quarter_decision = next((d for d in year_decisions if d["quarter"] == quarter), None)
                if quarter_decision:
                    budget = quarter_decision["budget"]
                    all_returns = {asset: self.market.returns_history[asset][quarter_idx] for asset in
                                   self.market.assets}
                    best_asset = max(all_returns.items(), key=lambda x: x[1])
                    max_return = budget * best_asset[1]
                    theoretical_max_return += max_return

        if total_investment > 0 and theoretical_max_return > 0:
            efficiency = (total_returns / theoretical_max_return) * 100
            formatted_output += f"  Investment Efficiency: {efficiency:.1f}% of maximum possible returns captured\n"
            formatted_output += f"  (Actual return: ${total_returns:.2f}, Maximum possible: ${theoretical_max_return:.2f})\n\n"

        # PART 2: HISTORICAL PERFORMANCE (if available)
        if current_year > 1:
            formatted_output += "\n## HISTORICAL PERFORMANCE COMPARISON\n\n"
            formatted_output += "Year-by-Year Performance:\n\n"

            # Table header
            formatted_output += "Year | Market | Budget Strategy | Growth | ROI | CEO Rating | Company Rating\n"
            formatted_output += "-----|--------|----------------|--------|-----|------------|---------------\n"

            # Add rows for each previous year
            for year in range(1, current_year):
                prev_market = self.market.market_conditions[year - 1]
                prev_year_calendar = 1970 + year

                # Get previous year budget
                prev_budget_info = self.budget[year - 1] if year <= len(self.budget) else None
                if prev_budget_info:
                    prev_total_percentage = prev_budget_info["percentage"] * 100
                    prev_distribution = prev_budget_info["distribution"]

                    if prev_distribution == "even":
                        if prev_total_percentage == 50:
                            prev_budget_str = "50%, even"
                        else:  # 100%
                            prev_budget_str = "100%, even"
                    else:  # increasing
                        if prev_total_percentage == 50:
                            prev_budget_str = "50%, increasing"
                        else:  # 100%
                            prev_budget_str = "100%, increasing"
                else:
                    prev_budget_str = "Unknown"

                # Calculate growth
                prev_start_idx = (year - 1) * 4
                prev_end_idx = year * 4

                if prev_start_idx < len(self.assets_logs) and prev_end_idx < len(self.assets_logs):
                    prev_starting_assets = self.assets_logs[prev_start_idx]
                    prev_ending_assets = self.assets_logs[prev_end_idx]
                    prev_growth_percentage = ((prev_ending_assets / prev_starting_assets) - 1) * 100
                else:
                    prev_growth_percentage = 0

                # Calculate ROI
                prev_year_decisions = [d for d in self.investment_decisions if d["year"] == year]
                prev_total_investment = sum(d["budget"] for d in prev_year_decisions)
                prev_total_returns = sum(d.get("return_amount", 0) for d in prev_year_decisions)

                if prev_total_investment > 0:
                    prev_return_percentage = (prev_total_returns / prev_total_investment) * 100
                else:
                    prev_return_percentage = 0

                # Get ratings if available
                ceo_rating = "N/A"
                company_rating = "N/A"

                if year - 1 < len(self.shareholder_ratings):
                    ratings = self.shareholder_ratings[year - 1]
                    if "ceo_rating" in ratings:
                        ceo_rating = f"{ratings['ceo_rating']:.1f}"
                    if "company_rating" in ratings:
                        company_rating = f"{ratings['company_rating']:.1f}"

                # Add row to table
                formatted_output += f"{year} ({prev_year_calendar}) | {prev_market.title()} | {prev_budget_str} | "
                formatted_output += f"{prev_growth_percentage:.1f}% | {prev_return_percentage:.1f}% | "
                formatted_output += f"{ceo_rating} | {company_rating}\n"

            # Add average historical performance (if multiple years)
            if current_year > 2:
                # Calculate averages
                avg_growth = sum(((self.assets_logs[year * 4] / self.assets_logs[(year - 1) * 4]) - 1) * 100
                                 for year in range(1, current_year)
                                 if (year - 1) * 4 < len(self.assets_logs) and year * 4 < len(self.assets_logs)) / (
                                         current_year - 1)

                avg_ceo_rating = 0
                avg_company_rating = 0
                rating_count = 0

                for year in range(1, current_year):
                    if year - 1 < len(self.shareholder_ratings):
                        ratings = self.shareholder_ratings[year - 1]
                        if "ceo_rating" in ratings and "company_rating" in ratings:
                            avg_ceo_rating += ratings["ceo_rating"]
                            avg_company_rating += ratings["company_rating"]
                            rating_count += 1

                if rating_count > 0:
                    avg_ceo_rating /= rating_count
                    avg_company_rating /= rating_count

                    formatted_output += f"Average | - | - | {avg_growth:.1f}% | - | {avg_ceo_rating:.1f} | {avg_company_rating:.1f}\n"

        return formatted_output

# %% md
# # Document
# %%
from concordia.document.document import Document
from concordia.document.interactive_document import InteractiveDocument

from concordia.document.document import Document
from concordia.document.interactive_document import InteractiveDocument


class MeetingDocumentManager:
    """Manages document-based meeting records for different agent perspectives with optimized context handling."""

    def __init__(self, model, embedder, company):
        """Initialize document manager with optimized context tracking."""
        self.model = model
        self.embedder = embedder
        self.company = company

        # Main document collections - one for each agent plus full record
        self.full_record = InteractiveDocument(model=self.model)

        # Agent context documents (keyed by agent name)
        self.agent_contexts = {}

        # Agent memory documents for persistent storage across meetings (keyed by agent name)
        self.agent_memories = {}

        # Initialize documents for each agent
        self._initialize_agent_documents()

        # Current meeting documents
        self.current_meeting_record = None
        self.current_agent_contexts = {}

        # Meeting summaries for reference (keyed by agent name)
        self.agent_meeting_summaries = {}

        # NEW: Track which rounds have been summarized for each meeting
        self.summarized_rounds = {}  # {meeting_id: set(round_numbers)}

        # NEW: Track current active round for each meeting
        self.current_round = {}  # {meeting_id: current_round_number}

        # NEW: Store round summaries
        self.round_summaries = {}  # {meeting_id: {agent_name: {round_num: {summary_type: content}}}}

    def _initialize_agent_documents(self):
        """Initialize document collections for each agent."""
        # Create context document for CEO
        self.agent_contexts[self.company.ceo.config.name] = InteractiveDocument(model=self.model)
        self.agent_contexts[self.company.ceo.config.name].statement(
            f"# {self.company.ceo.config.name}'s Meeting Records\n\n")

        # Create memory document for CEO
        self.agent_memories[self.company.ceo.config.name] = InteractiveDocument(model=self.model)
        self.agent_memories[self.company.ceo.config.name].statement(
            f"# {self.company.ceo.config.name}'s Meeting Memories\n\n")

        # Create context and memory documents for each shareholder
        for shareholder_name, shareholder in self.company.shareholders.items():
            # Context document for meeting records
            self.agent_contexts[shareholder_name] = InteractiveDocument(model=self.model)
            self.agent_contexts[shareholder_name].statement(f"# {shareholder_name}'s Meeting Records\n\n")

            # Memory document for persistent memories
            self.agent_memories[shareholder_name] = InteractiveDocument(model=self.model)
            self.agent_memories[shareholder_name].statement(f"# {shareholder_name}'s Meeting Memories\n\n")

        # Initialize full record with title
        self.full_record.statement("# Full Meeting Records\n\n")

        # Initialize meeting summaries for each agent
        self.agent_meeting_summaries = {}
        for agent_name in self.agent_contexts.keys():
            self.agent_meeting_summaries[agent_name] = InteractiveDocument(model=self.model)
            self.agent_meeting_summaries[agent_name].statement(f"# {agent_name}'s Meeting Summaries\n\n")

    def initialize_meeting(self, meeting_id, meeting_type, year, quarter=None):
        """Initialize meeting documents with reflection-based memory transfer."""
        title_suffix = f" Year {year}" + (f" Q{quarter}" if quarter else "")

        # Initialize round tracking for this meeting
        self.summarized_rounds[meeting_id] = set()
        self.current_round[meeting_id] = 0  # Start at round 0

        # Initialize summaries dictionary for this meeting
        self.round_summaries[meeting_id] = {}

        # Create fresh document for full meeting record
        self.current_meeting_record = InteractiveDocument(model=self.model)

        # Create fresh documents for each agent
        self.current_agent_contexts = {}

        # Standard header for all documents
        header = f"# {meeting_type.replace('_', ' ').title()}{title_suffix}\n\n"
        self.current_meeting_record.statement(header)

        # Add headers to all agent documents
        for agent_name in self.agent_contexts.keys():
            self.current_agent_contexts[agent_name] = InteractiveDocument(model=self.model)
            self.current_agent_contexts[agent_name].statement(header)

            # Initialize summaries dictionary for this agent
            self.round_summaries[meeting_id][agent_name] = {}

            # Get only reflections from previous meetings, not all memories
            relevant_reflections = self._get_relevant_memories(agent_name, meeting_type, year, quarter)
            if relevant_reflections:
                reflection_section = f"## Previous Meeting Reflections\n\n{relevant_reflections}\n\n"
                self.current_agent_contexts[agent_name].statement(reflection_section)

                # Also add a separator to clearly distinguish memory from current meeting
                self.current_agent_contexts[agent_name].statement("## Current Meeting\n\n")

        return {
            "full": self.current_meeting_record,
            "agents": self.current_agent_contexts
        }

    def _get_relevant_memories(self, agent_name, meeting_type, year, quarter=None):
        """Retrieve only meeting reflections relevant to an agent for a particular meeting."""
        if agent_name not in self.agent_memories:
            return ""

        memory_doc = self.agent_memories[agent_name]

        # Use view to get all meeting reflections
        reflection_view = memory_doc.view(include_tags=["every_meeting_reflection"])
        reflection_text = reflection_view.text()

        # If no reflections found
        if not reflection_text or reflection_text.strip() == "":
            return f"No previous meeting reflections found for {meeting_type.replace('_', ' ')} Year {year}" + (
                f" Quarter {quarter}" if quarter else "")

        # Return all reflections - they're already organized chronologically
        return reflection_text

    def add_to_all_documents(self, text, tags=None):
        """Add content to all agent documents with appropriate tagging."""
        if tags is None:
            tags = []

        # Add to full meeting record
        self.full_record.statement(text, tags=tags)

        # Add to each agent's document if current_meeting_record is initialized
        if self.current_meeting_record is not None:
            self.current_meeting_record.statement(text, tags=tags)

        # Add to each agent's context document if initialized
        for agent_name, doc in self.current_agent_contexts.items():
            if doc is not None:
                doc.statement(text, tags=tags)

    def add_to_agent_only(self, agent_name, text, tags=None):
        """Add content only to a specific agent's document."""
        if tags is None:
            tags = []

        # Add to full meeting record for comprehensive logging
        self.current_meeting_record.statement(text, tags=tags)

        # Add only to the specific agent's document
        if agent_name in self.current_agent_contexts:
            self.current_agent_contexts[agent_name].statement(text, tags=tags)

    def add_to_specific_documents(self, agent_names, text, tags=None):
        """Add content to specific agents' documents and the master record."""
        if tags is None:
            tags = []

        # Add to full meeting record
        self.full_record.statement(text, tags=tags)

        # Add to current meeting record if initialized
        if self.current_meeting_record is not None:
            self.current_meeting_record.statement(text, tags=tags)

        # Add to specified agents' documents
        for agent_name in agent_names:
            if agent_name in self.current_agent_contexts:
                self.current_agent_contexts[agent_name].statement(text, tags=tags)

    def add_to_agent_memory(self, agent_name, text, tags=None):
        """Add content to an agent's persistent memory document."""
        if tags is None:
            tags = []

        # Add to the agent's memory document if it exists
        if agent_name in self.agent_memories:
            self.agent_memories[agent_name].statement(text, tags=tags)

    def get_agent_context(self, agent_name, filter_summarized_rounds=True):
        """Get agent context with optional filtering of summarized rounds."""
        # If agent_name in current_agent_contexts, use that document
        if agent_name in self.current_agent_contexts:
            doc = self.current_agent_contexts[agent_name]

            # If filtering is requested and we have rounds to filter
            if filter_summarized_rounds and self.summarized_rounds:
                exclude_tags = []

                # Create exclude tags for all summarized rounds
                for meeting_id, rounds in self.summarized_rounds.items():
                    for round_num in rounds:
                        exclude_tags.append(f"{meeting_id}_round{round_num}_filterable")

                # Create filtered view if we have tags to exclude
                if exclude_tags:
                    filtered_view = doc.view(exclude_tags=exclude_tags)
                    return filtered_view.text()

            # Return full text if no filtering requested or no tags to exclude
            return doc.text()

        return ""  # Return empty string if agent document not found
    def get_agent_memory(self, agent_name):
        """Get an agent's memory document text."""
        if agent_name in self.agent_memories:
            return self.agent_memories[agent_name].text()
        return ""

    def generate_ceo_round_summaries(self, meeting_id, current_round, ceo_name):
        """Generate CEO-specific summaries for the specified round, focused on leadership aspects."""
        # Skip if already summarized
        if meeting_id in self.round_summaries and ceo_name in self.round_summaries[meeting_id] and \
                current_round in self.round_summaries[meeting_id][ceo_name]:
            return self.round_summaries[meeting_id][ceo_name][current_round]

        # Create ID for this document
        summary_doc_id = f"{meeting_id}_round{current_round}_{ceo_name}_summary"

        # Get the CEO's complete context without filtering
        full_context = self.get_agent_context(ceo_name, filter_summarized_rounds=False)

        # Create interactive document for generating summaries
        summary_doc = InteractiveDocument(model=self.model)

        # Generate the four summary types appropriate for CEO

        # 1. Discussion flow summary
        discussion_flow_prompt = f"""As the CEO, summarize the overall flow of discussion in round {current_round}.

    Focus on:
    - How the discussion progressed in relation to your planned theme
    - Key turning points or insights that emerged
    - Level of engagement from shareholders
    - How well the discussion addressed the intended objectives

    Keep your summary concise (3-5 sentences) focused on the effectiveness of the discussion process.
    """

        discussion_flow_summary = summary_doc.open_question(
            discussion_flow_prompt + "\n\n" + full_context,
            answer_label="Discussion Flow Summary",
            max_tokens=200
        )

        # 2. Shareholder positions summary
        shareholder_positions_prompt = f"""As the CEO, summarize the key positions taken by shareholders during round {current_round}.

    Focus on:
    - Which options different shareholders supported
    - Main arguments presented for each option
    - Areas of consensus or disagreement among shareholders
    - How shareholder positions align with or differ from your perspective

    Maintain an objective assessment of the different viewpoints expressed.
    """

        shareholder_positions_summary = summary_doc.open_question(
            shareholder_positions_prompt + "\n\n" + full_context,
            answer_label="Shareholder Positions Summary",
            max_tokens=200
        )

        # 3. Decision progress analysis
        decision_progress_prompt = f"""As the CEO, analyze the progress made toward a decision in round {current_round}.

    Focus on:
    - How far the group has moved toward consensus
    - Which options are gaining or losing support
    - What key concerns or objections remain unaddressed
    - What additional information or discussion might be needed

    Provide a strategic assessment of where the decision process stands.
    """

        decision_progress_summary = summary_doc.open_question(
            decision_progress_prompt + "\n\n" + full_context,
            answer_label="Decision Progress Analysis",
            max_tokens=250
        )

        # 4. Leadership strategy assessment
        leadership_strategy_prompt = f"""As the CEO, reflect on your leadership approach during round {current_round} and plan for next steps.

    Consider:
    - How effective your contributions were in guiding the discussion
    - Whether your position is gaining support or needs adjustment
    - How well you addressed shareholder concerns
    - What leadership approach would be most effective in the next round

    This is your private strategic assessment to guide your leadership.
    """

        leadership_strategy_summary = summary_doc.open_question(
            leadership_strategy_prompt + "\n\n" + full_context,
            answer_label="Leadership Strategy Assessment",
            max_tokens=200
        )

        # Store summaries in dictionary
        summaries = {
            "discussion_flow": discussion_flow_summary,
            "shareholder_positions": shareholder_positions_summary,
            "decision_progress": decision_progress_summary,
            "leadership_strategy": leadership_strategy_summary
        }

        # Store in round_summaries dictionary
        if meeting_id not in self.round_summaries:
            self.round_summaries[meeting_id] = {}
        if ceo_name not in self.round_summaries[meeting_id]:
            self.round_summaries[meeting_id][ceo_name] = {}
        self.round_summaries[meeting_id][ceo_name][current_round] = summaries

        # Add combined summary to CEO document and memory
        combined_summary = f"## Round {current_round} Discussion Summary\n\n"
        combined_summary += f"### Discussion Flow\n{discussion_flow_summary}\n\n"
        combined_summary += f"### Shareholder Positions\n{shareholder_positions_summary}\n\n"
        combined_summary += f"### Decision Progress\n{decision_progress_summary}\n\n"
        combined_summary += f"### Leadership Strategy\n{leadership_strategy_summary}\n\n"

        # Add to CEO's document with appropriate tags
        if ceo_name in self.current_agent_contexts:
            summary_tags = [f"round_{current_round}_summary", "discussion_summary"]
            self.add_to_agent_only(ceo_name, combined_summary, tags=summary_tags)

        # Add to CEO's memory for long-term reference
        memory_text = f"## Round {current_round} Discussion Summary for {meeting_id}\n\n"
        memory_text += combined_summary
        self.add_to_agent_memory(ceo_name, memory_text, tags=["discussion_summary", meeting_id])

        return summaries

    def generate_round_summaries(self, meeting_id, current_round, agent_name):
        """Generate summaries for the specified round from an agent's perspective."""
        # Check if this agent is the CEO and use the CEO-specific function if so
        if hasattr(self.company, 'ceo') and agent_name == self.company.ceo.config.name:
            return self.generate_ceo_round_summaries(meeting_id, current_round, agent_name)

        # Skip if already summarized
        if meeting_id in self.round_summaries and agent_name in self.round_summaries[meeting_id] and \
                current_round in self.round_summaries[meeting_id][agent_name]:
            return self.round_summaries[meeting_id][agent_name][current_round]

        # Create ID for this document
        summary_doc_id = f"{meeting_id}_round{current_round}_{agent_name}_summary"

        # Get the agent's complete context without filtering
        full_context = self.get_agent_context(agent_name, filter_summarized_rounds=False)

        # Create interactive document for generating summaries
        summary_doc = InteractiveDocument(model=self.model)

        # Get CEO name for clear reference
        ceo_name = self.company.ceo.config.name if hasattr(self.company, 'ceo') else "CEO"

        # Generate the four summary types

        # 1. Self-CEO dialogue summary
        self_dialogue_prompt = f"""As {agent_name}, summarize only your direct dialogue with the CEO in discussion round {current_round}.

Focus on:
- What positions or arguments you personally made
- How the CEO responded to your points
- Any agreements or disagreements between you and the CEO

Keep your summary concise (3-5 sentences) and focused on the substance of the exchange.
"""

        self_dialogue_summary = summary_doc.open_question(
            self_dialogue_prompt + "\n\n" + full_context,
            answer_label="Self-CEO Dialogue Summary",
            max_tokens=200
        )

        # 2. Other shareholders' contributions summary
        others_summary_prompt = f"""As {agent_name}, summarize the key contributions of other shareholders (not yourself or the CEO) during discussion round {current_round}.

Focus on:
- Which positions or options different shareholders supported
- Main arguments or evidence they presented
- Notable agreements or disagreements among shareholders

Keep your summary concise (3-5 sentences) and focused on the substance of their contributions.
"""

        others_summary = summary_doc.open_question(
            others_summary_prompt + "\n\n" + full_context,
            answer_label="Other Shareholders' Contributions Summary",
            max_tokens=200
        )

        # 3. Shareholder position analysis
        positions_analysis_prompt = f"""As {agent_name}, analyze what position each participant appears to support based on round {current_round} discussions.

Important: Clearly distinguish between the CEO and shareholders in your analysis.
- For the CEO ({ceo_name}): Note their apparent position
- For each shareholder (including yourself): Note their apparent position
- For each person, indicate confidence in your assessment (certain, likely, unclear)
- Base your analysis only on what was explicitly stated

Format as a brief list with one line per person, properly labeled as either "CEO" or the shareholder's name. Keep the entire analysis under 200 words.
"""

        positions_summary = summary_doc.open_question(
            positions_analysis_prompt + "\n\n" + full_context,
            answer_label="Shareholder Positions Analysis",
            max_tokens=250
        )

        # 4. CEO position analysis
        ceo_analysis_prompt = f"""As {agent_name}, analyze the CEO's position based on their statements in round {current_round}.

Focus on:
- Which option the CEO ({ceo_name}) appears to favor
- Their key arguments or reasoning
- How their position relates to shareholder input
- Whether their position has evolved during the discussion

Keep your analysis concise (3-5 sentences) and based only on what was explicitly stated.
"""

        ceo_summary = summary_doc.open_question(
            ceo_analysis_prompt + "\n\n" + full_context,
            answer_label="CEO Position Analysis",
            max_tokens=200
        )

        # Store summaries in dictionary
        summaries = {
            "self_dialogue": self_dialogue_summary,
            "others_contributions": others_summary,
            "positions_analysis": positions_summary,
            "ceo_analysis": ceo_summary
        }

        # Store in round_summaries dictionary
        if meeting_id not in self.round_summaries:
            self.round_summaries[meeting_id] = {}
        if agent_name not in self.round_summaries[meeting_id]:
            self.round_summaries[meeting_id][agent_name] = {}
        self.round_summaries[meeting_id][agent_name][current_round] = summaries

        # Add combined summary to agent document and memory
        combined_summary = f"## Round {current_round} Discussion Summary\n\n"
        combined_summary += f"### My Dialogue with CEO\n{self_dialogue_summary}\n\n"
        combined_summary += f"### Other Shareholders' Contributions\n{others_summary}\n\n"
        combined_summary += f"### Analysis of Shareholder Positions\n{positions_summary}\n\n"
        combined_summary += f"### Analysis of CEO's Position\n{ceo_summary}\n\n"

        # Add to agent's document with appropriate tags
        if agent_name in self.current_agent_contexts:
            summary_tags = [f"round_{current_round}_summary", "discussion_summary"]
            self.add_to_agent_only(agent_name, combined_summary, tags=summary_tags)

        # Add to agent's memory for long-term reference
        memory_text = f"## Round {current_round} Discussion Summary for {meeting_id}\n\n"
        memory_text += combined_summary
        self.add_to_agent_memory(agent_name, memory_text, tags=["discussion_summary", meeting_id])

        return summaries

    def update_round_tracking(self, meeting_id, current_round):
        """Update current round tracking for a meeting."""
        self.current_round[meeting_id] = current_round

    def finalize_meeting(self):
        """Finalize meeting documents and generate meeting reflections."""
        # Create meeting reflections for each agent in parallel
        reflections = {}
        meeting_id = self.current_meeting_record.text().split("\n")[0].strip("# ")
        # Define worker function for parallel reflection generation
        def _reflection_worker(agent_item):
            agent_name, doc = agent_item

            # Get the corresponding agent object
            if hasattr(self.company, 'ceo') and agent_name == self.company.ceo.config.name:
                agent = self.company.ceo
            elif agent_name in self.company.shareholders:
                agent = self.company.shareholders[agent_name]
            else:
                return agent_name, None  # Skip if agent not found

            # Generate reflection
            reflection = self.generate_meeting_reflection(agent, doc)

            # Add reflection to agent's memory document with appropriate tags
            self.add_to_agent_memory(
                agent_name,
                f"## Reflection: {meeting_id}\n\n{reflection}\n\n",
                tags=["every_meeting_reflection", meeting_id]
            )

            return agent_name, reflection

        # Process all agents in parallel
        agent_items = list(self.current_agent_contexts.items())

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(agent_items)) as executor:
            for agent_name, reflection in executor.map(_reflection_worker, agent_items):
                if reflection is not None:
                    reflections[agent_name] = reflection
        # time.sleep(5)
        # Add complete documents to the master collections
        self.full_record.statement(self.current_meeting_record.text())

        # This part still needs to be sequential due to potential conflicts
        for agent_name, doc in self.current_agent_contexts.items():
            if agent_name in self.agent_contexts:
                self.agent_contexts[agent_name].statement(doc.text())

        # NEW: Save meeting results
        if hasattr(self.company, 'simulation') and hasattr(self.company.simulation, 'result_saver'):
            # If the company has access to the simulation object with the saver
            ceo_mbti= self.company.ceo.config.traits if hasattr(self.company.ceo.config, 'traits') else "Unknown"
            self.company.simulation.result_saver.save_meeting_results(
                self.company.simulation.simulation_folder,
                meeting_id,
                self
            )

        return {
            "reflections": reflections
        }

    def generate_meeting_reflection(self, agent, doc):
        """Generate a structured reflection for an agent based on the meeting document."""
        # Extract basic agent information
        agent_name = agent.config.name
        mbti_type = agent.config.traits if hasattr(agent.config, 'traits') else "Unknown"
        is_ceo = agent_name == self.company.ceo.config.name
        role = "CEO" if is_ceo else "Shareholder"

        # Extract meeting ID from document
        meeting_id_match = re.search(r'(annual_budget_year\d+|quarterly_investment_year\d+_q\d+|annual_review_year\d+)',
                                     doc.text())
        meeting_id = meeting_id_match.group(1) if meeting_id_match else "unknown_meeting"

        # Get position and voting information using existing manager functions
        position_info = ""
        voting_info = ""

        # Get position summary if available
        if hasattr(self, 'position_manager'):
            try:
                position_info = self.position_manager.get_position_summary_text(meeting_id, agent_name)
            except:
                position_info = "No position data available."

        # Get voting summary if available
        if hasattr(self, 'voting_manager'):
            for prop_type in ["initial", "alternative"]:
                if self.voting_manager.has_votes(meeting_id, prop_type):
                    try:
                        vote_summary = self.voting_manager.get_agent_vote_summary(meeting_id, agent_name, prop_type)
                        voting_info += vote_summary + "\n"
                    except:
                        pass

            if not voting_info:
                voting_info = "No voting data available."

        # Generate reflection with three focused points
        reflection_prompt = f"""As {agent_name} ({role}) with MBTI type {mbti_type}, create a structured reflection on this meeting.

        Please provide exactly 3 sections with detailed content:

        1. IMPORTANT EVENTS: Record the key events from this meeting from your perspective, including:
           - Your decisions and positions in the meeting
           - How others (especially CEO) responded to your input
           - Voting outcomes and the final company decision
           - Whether your preferred option was adopted

        2. CEO & COMPANY EVALUATION: Provide specific evaluations based on concrete events:
           - What the CEO did well (cite specific examples)
           - What the CEO could improve (cite specific examples)
           - How the company's decision-making process worked this meeting
           - Whether you think the final decision was good for the company

        3. LESSONS & REFLECTIONS: Share insights for future meetings:
           - What you learned about the decision-making dynamics
           - How you might adjust your approach in future meetings
           - Any patterns you noticed that could help predict outcomes
           - Key takeaways that will influence your future participation

        Provide detailed content for each section. Be specific and reference actual events, not abstract concepts.

        Meeting Context:
        {doc.text()}

        Your Position History:
        {position_info}

        Your Voting Information:
        {voting_info}
        """

        try:
            # Use standard text generation with increased token limit
            reflection = self.model.sample_text(
                prompt=reflection_prompt,
                max_tokens=800,
                temperature=0.7
            )
        except Exception as e:
            reflection = f"Error generating reflection: {str(e)}"

        # Check if reflection is empty and provide default message
        if not reflection or reflection.strip() == "":
            reflection = f"No reflection was generated for this meeting ({meeting_id})."

        return reflection

    def update_filter_for_round(self, meeting_id, round_num):
        """Update filters to exclude this round's content without generating summaries."""
        # Make sure meeting_id matches the format used in run_public_discussion
        # (remove "_meeting" if present to match document tag format)
        standard_meeting_id = meeting_id.replace("_meeting", "") if "_meeting" in meeting_id else meeting_id

        # Update tracking of summarized rounds
        if standard_meeting_id not in self.summarized_rounds:
            self.summarized_rounds[standard_meeting_id] = set()
        self.summarized_rounds[standard_meeting_id].add(round_num)

        # Log the update
        print(f"Updated filters for meeting {standard_meeting_id}, round {round_num}")

    def get_confidence_data_safely(self, position_entry, default=None):
        """Safely extract confidence distribution from a position entry."""
        if not position_entry:
            return default or {}

        # Try to get confidence distribution with fallback to empty dict
        return position_entry.get('confidence_distribution', default or {})



class PositionManager:
    """Manages position tracking for all agents across meetings with confidence distribution support."""

    def __init__(self):
        """Initialize an empty position manager."""
        self.positions = {}  # {meeting_id: {agent_name: [position_entries]}}

    def initialize_meeting(self, meeting_id):
        """Initialize position storage for a new meeting."""
        if meeting_id not in self.positions:
            self.positions[meeting_id] = {}

    def record_position(self, meeting_id, agent_name, round_num, option, reasoning, confidence_distribution=None,
                        changed=False):
        self.initialize_meeting(meeting_id)

        if agent_name not in self.positions[meeting_id]:
            self.positions[meeting_id][agent_name] = []

        position_entry = {
            'round': round_num,
            'option': option,
            'reasoning': reasoning,
            'confidence_distribution': confidence_distribution or {},  # Store confidence for each option
            'changed': changed
        }

        self.positions[meeting_id][agent_name].append(position_entry)
        return position_entry

    def get_current_position(self, meeting_id, agent_name):
        """Get agent's current position for a meeting."""
        if not self._check_position_exists(meeting_id, agent_name):
            return None

        # Ensure backward compatibility
        self.ensure_backward_compatibility(meeting_id, agent_name)

        positions = self.positions[meeting_id][agent_name]
        return positions[-1] if positions else None

    def get_initial_position(self, meeting_id, agent_name):
        """Get agent's initial position for a meeting."""
        if not self._check_position_exists(meeting_id, agent_name):
            return None

        # Ensure backward compatibility
        self.ensure_backward_compatibility(meeting_id, agent_name)

        positions = self.positions[meeting_id][agent_name]
        return positions[0] if positions else None

    def get_position_history(self, meeting_id, agent_name):
        """Get full position history for an agent in a meeting with backward compatibility."""
        if not self._check_position_exists(meeting_id, agent_name):
            return []

        # Ensure backward compatibility
        self.ensure_backward_compatibility(meeting_id, agent_name)

        return self.positions[meeting_id][agent_name]

    def get_confidence_distribution(self, meeting_id, agent_name, round_num=None):
        """Get confidence distribution, safely handling old position data format."""
        if not self._check_position_exists(meeting_id, agent_name):
            return {}

        # Ensure backward compatibility
        self.ensure_backward_compatibility(meeting_id, agent_name)

        positions = self.positions[meeting_id][agent_name]

        if round_num is not None:
            # Find position for specific round
            for position in positions:
                if position.get('round') == round_num:
                    return position.get('confidence_distribution', {})
            return {}  # Round not found
        else:
            # Return latest confidence distribution
            return positions[-1].get('confidence_distribution', {}) if positions else {}

    def get_confidence_evolution(self, meeting_id, agent_name):
        if not self._check_position_exists(meeting_id, agent_name):
            return {}

        evolution = {}
        for position in self.positions[meeting_id][agent_name]:
            round_num = position.get('round', 0)
            confidence = position.get('confidence_distribution', {})
            if confidence:  # Only include entries that have confidence data
                evolution[round_num] = confidence

        return evolution

    def calculate_confidence_shifts(self, meeting_id, agent_name):
        """Calculate confidence shifts with backward compatibility support."""
        if not self._check_position_exists(meeting_id, agent_name):
            return []

        # Ensure backward compatibility
        self.ensure_backward_compatibility(meeting_id, agent_name)

        positions = self.positions[meeting_id][agent_name]
        if len(positions) < 2:
            return []

        shifts = []
        all_options = set()

        # Collect all options that appear in any confidence distribution
        for position in positions:
            confidence = position.get('confidence_distribution', {})
            all_options.update(confidence.keys())

        # Calculate shifts between consecutive positions
        for i in range(1, len(positions)):
            prev_confidence = positions[i - 1].get('confidence_distribution', {})
            curr_confidence = positions[i].get('confidence_distribution', {})

            # Skip if either doesn't have confidence data
            if not prev_confidence or not curr_confidence:
                continue

            round_shifts = {
                'round': positions[i].get('round', i),
                'shifts': {}
            }

            # Calculate shifts for each option
            for option in all_options:
                prev_value = prev_confidence.get(option, 0)
                curr_value = curr_confidence.get(option, 0)
                shift = curr_value - prev_value

                if shift != 0:  # Only include non-zero shifts
                    round_shifts['shifts'][option] = shift

            # Only add if there were any shifts
            if round_shifts['shifts']:
                # Identify key influencing arguments
                if positions[i].get('reasoning'):
                    round_shifts['key_argument'] = positions[i].get('reasoning')

                shifts.append(round_shifts)

        return shifts

    def has_position_changed(self, meeting_id, agent_name):
        """Check if agent's position changed during the meeting - works with both formats."""
        if not self._check_position_exists(meeting_id, agent_name):
            return False

        positions = self.positions[meeting_id][agent_name]
        if len(positions) < 2:
            return False

        # Just compare the option field directly - works for both formats
        return positions[0]['option'] != positions[-1]['option']

    def count_position_changes(self, meeting_id, agent_name):
        """Count how many times an agent changed positions during a meeting."""
        if not self._check_position_exists(meeting_id, agent_name):
            return 0

        return sum(1 for pos in self.positions[meeting_id][agent_name] if pos.get('changed', False))

    def get_agreement_rate(self, meeting_id, option):
        """Calculate percentage of agents who support a specific option."""
        if meeting_id not in self.positions:
            return 0.0

        total_agents = len(self.positions[meeting_id])
        if total_agents == 0:
            return 0.0

        agreements = sum(1 for agent in self.positions[meeting_id]
                         if self.get_current_position(meeting_id, agent)['option'] == option)

        return agreements / total_agents

    def get_position_summary(self, meeting_id):
        """Generate summary of all positions and changes for a meeting with backward compatibility."""
        if meeting_id not in self.positions:
            return {
                "ceo": {"changes": 0, "summary": "No position data available"},
                "shareholders": {},
                "overall": "No position tracking data available for this meeting"
            }

        # Apply backward compatibility to all agents
        for agent_name in self.positions[meeting_id].keys():
            self.ensure_backward_compatibility(meeting_id, agent_name)

        summary = {
            "ceo": {},
            "shareholders": {},
            "overall": ""
        }

        # Identify CEO by convention - this could be made more robust
        ceo_name = next((name for name in self.positions[meeting_id]
                         if name.lower().endswith("(ceo)") or "ceo" in name.lower()), None)

        # Process CEO data if found
        if ceo_name:
            ceo_data = self.positions[meeting_id][ceo_name]
            ceo_initial = ceo_data[0]['option'] if ceo_data else "Unknown"
            ceo_current = ceo_data[-1]['option'] if ceo_data else "Unknown"
            ceo_changes = self.count_position_changes(meeting_id, ceo_name)

            # Get confidence distributions if available
            ceo_initial_confidence = ceo_data[0].get('confidence_distribution', {}) if ceo_data else {}
            ceo_final_confidence = ceo_data[-1].get('confidence_distribution', {}) if ceo_data else {}

            change_details = []
            for entry in ceo_data:
                if entry.get('changed', False):
                    change_detail = {
                        "round": entry['round'],
                        "new_position": entry['option'],
                        "reasoning": entry['reasoning']
                    }
                    # Add confidence if available
                    if 'confidence_distribution' in entry:
                        change_detail["confidence"] = entry['confidence_distribution']

                    change_details.append(change_detail)

            summary["ceo"] = {
                "name": ceo_name,
                "initial_position": ceo_initial,
                "final_position": ceo_current,
                "changes": ceo_changes,
                "change_details": change_details,
                "summary": f"CEO began favoring {ceo_initial} and ended with {ceo_current}, making {ceo_changes} change(s)"
            }

            # Add confidence information if available
            if ceo_initial_confidence:
                summary["ceo"]["initial_confidence"] = ceo_initial_confidence
            if ceo_final_confidence:
                summary["ceo"]["final_confidence"] = ceo_final_confidence

            # Add confidence shifts
            confidence_shifts = self.calculate_confidence_shifts(meeting_id, ceo_name)
            if confidence_shifts:
                summary["ceo"]["confidence_shifts"] = confidence_shifts

        # Process shareholder data
        shareholders = [name for name in self.positions[meeting_id] if name != ceo_name]
        total_shareholder_changes = 0

        for name in shareholders:
            positions = self.positions[meeting_id][name]
            if not positions:
                continue

            initial_position = positions[0]['option']
            current_position = positions[-1]['option']
            position_changes = self.count_position_changes(meeting_id, name)
            total_shareholder_changes += position_changes

            # Get confidence distributions if available
            initial_confidence = positions[0].get('confidence_distribution', {})
            final_confidence = positions[-1].get('confidence_distribution', {})

            change_details = []
            for entry in positions:
                if entry.get('changed', False):
                    change_detail = {
                        "round": entry['round'],
                        "new_position": entry['option'],
                        "reasoning": entry['reasoning']
                    }
                    # Add confidence if available
                    if 'confidence_distribution' in entry:
                        change_detail["confidence"] = entry['confidence_distribution']

                    change_details.append(change_detail)

            shareholder_summary = {
                "initial_position": initial_position,
                "final_position": current_position,
                "changes": position_changes,
                "change_details": change_details,
                "summary": f"Started with {initial_position} and ended with {current_position}, making {position_changes} change(s)"
            }

            # Add confidence information if available
            if initial_confidence:
                shareholder_summary["initial_confidence"] = initial_confidence
            if final_confidence:
                shareholder_summary["final_confidence"] = final_confidence

            # Add confidence shifts
            confidence_shifts = self.calculate_confidence_shifts(meeting_id, name)
            if confidence_shifts:
                shareholder_summary["confidence_shifts"] = confidence_shifts

            summary["shareholders"][name] = shareholder_summary

        # Calculate agreement stats
        if ceo_name and ceo_data:
            initial_agreements = 0
            final_agreements = 0

            for shareholder in shareholders:
                sh_positions = self.positions[meeting_id][shareholder]
                if not sh_positions:
                    continue

                if sh_positions[0]['option'] == ceo_initial:
                    initial_agreements += 1
                if sh_positions[-1]['option'] == ceo_current:
                    final_agreements += 1

            shareholder_count = len(shareholders)
            if shareholder_count > 0:
                initial_agreement_rate = initial_agreements / shareholder_count
                final_agreement_rate = final_agreements / shareholder_count

                summary[
                    "overall"] = f"Discussion started with {initial_agreement_rate * 100:.1f}% agreement with CEO ({ceo_initial}) "
                summary["overall"] += f"and ended with {final_agreement_rate * 100:.1f}% agreement ({ceo_current}). "

                total_participants = 1 + shareholder_count
                total_changes = total_shareholder_changes + ceo_changes
                avg_changes = total_changes / total_participants
                summary["overall"] += f"Participants made an average of {avg_changes:.1f} position changes."

        return summary

    def is_ceo(self, agent_name):
        """Helper method to determine if an agent is the CEO."""
        return "ceo" in agent_name.lower() or agent_name.lower().endswith("(ceo)")

    def _check_position_exists(self, meeting_id, agent_name):
        """Helper method to check if positions exist for meeting and agent."""
        return (meeting_id in self.positions and
                agent_name in self.positions[meeting_id] and
                len(self.positions[meeting_id][agent_name]) > 0)

    def get_position_summary_text(self, meeting_id, agent_name):
        """Generate a formatted text summary of an agent's positions for a meeting."""
        # Check if we have position data for this meeting and agent
        if not self._check_position_exists(meeting_id, agent_name):
            return f"# Position Summary for {agent_name} in {meeting_id}\n\nNo position data available."

        # Get position history
        position_history = self.get_position_history(meeting_id, agent_name)

        # Extract meeting details for a more user-friendly title
        meeting_type = "Discussion"
        year = None
        quarter = None

        # Parse meeting_id to extract details
        if "annual_budget" in meeting_id:
            meeting_type = "Annual Budget Meeting"
        elif "quarterly_investment" in meeting_id:
            meeting_type = "Quarterly Investment Meeting"
            quarter_match = re.search(r'q(\d+)', meeting_id)
            if quarter_match:
                quarter = int(quarter_match.group(1))
        elif "annual_review" in meeting_id:
            meeting_type = "Annual Review Meeting"

        year_match = re.search(r'year(\d+)', meeting_id)
        if year_match:
            year = int(year_match.group(1))

        # Create a title with meeting details
        title = f"# Position Summary for {agent_name}"
        if year:
            title += f" - Year {year}"
        if quarter:
            title += f", Quarter {quarter}"
        title += f" {meeting_type}"

        # Start building the summary
        summary = f"{title}\n\n"

        # Add initial position
        initial_position = position_history[0] if position_history else None
        if initial_position:
            summary += f"## Initial Position\n"
            summary += f"Position: {initial_position['option']}\n"

            # Add confidence distribution if available
            initial_confidence = initial_position.get('confidence_distribution', {})
            if initial_confidence:
                summary += "Confidence Distribution:\n"
                for option, confidence in initial_confidence.items():
                    summary += f"- {option}: {confidence * 100:.1f}%\n"

            summary += f"Reasoning: {initial_position['reasoning']}\n\n"

        # Add subsequent position changes
        changes_found = False
        if len(position_history) > 1:
            # Skip the initial position (index 0) as we've already included it
            for i, position in enumerate(position_history[1:], start=1):
                # Only include positions that represent changes or are after discussion rounds
                round_num = position.get('round', 0)
                if round_num > 0:  # Only include positions after discussion rounds
                    if not changes_found:
                        summary += f"## Position Evolution\n"
                        changes_found = True

                    summary += f"### After Round {round_num}\n"
                    summary += f"Position: {position['option']}\n"

                    # Add confidence distribution if available
                    confidence = position.get('confidence_distribution', {})
                    if confidence:
                        summary += "Confidence Distribution:\n"
                        for option, conf_value in confidence.items():
                            summary += f"- {option}: {conf_value * 100:.1f}%\n"

                        # Add confidence shifts compared to previous round
                        if i > 0 and 'confidence_distribution' in position_history[i - 1]:
                            prev_confidence = position_history[i - 1]['confidence_distribution']
                            summary += "Confidence Shifts:\n"
                            for option, curr_conf in confidence.items():
                                prev_conf = prev_confidence.get(option, 0)
                                shift = curr_conf - prev_conf
                                if shift != 0:
                                    summary += f"- {option}: {'+' if shift > 0 else ''}{shift * 100:.1f}%\n"

                    # Indicate if this was a change
                    if position.get('changed', False):
                        summary += f"**Changed position**\n"
                    else:
                        summary += f"*Maintained position*\n"

                    summary += f"Reasoning: {position['reasoning']}\n\n"

        # Add overall summary
        has_changed = self.has_position_changed(meeting_id, agent_name)
        change_count = self.count_position_changes(meeting_id, agent_name)

        summary += f"## Summary\n"
        current_position = position_history[-1]['option'] if position_history else "Unknown"

        if has_changed:
            initial_position = position_history[0]['option'] if position_history else "Unknown"
            summary += f"Starting with {initial_position}, {agent_name} changed positions {change_count} time(s) "
            summary += f"and currently supports {current_position}.\n"
        else:
            summary += f"{agent_name} has consistently supported {current_position} throughout the discussion.\n"

        # Add overall confidence evolution if available
        confidence_evolution = self.get_confidence_evolution(meeting_id, agent_name)
        if len(confidence_evolution) > 1:  # Only if we have multiple rounds with confidence
            summary += "\n## Confidence Evolution\n\n"

            # Find all options across all rounds
            all_options = set()
            for conf_dist in confidence_evolution.values():
                all_options.update(conf_dist.keys())

            # Display evolution for each option
            for option in sorted(all_options):
                summary += f"### {option}\n"
                summary += "Round | Confidence\n"
                summary += "------|------------\n"

                for round_num in sorted(confidence_evolution.keys()):
                    conf_value = confidence_evolution[round_num].get(option, 0)
                    summary += f"{round_num} | {conf_value * 100:.1f}%\n"

                summary += "\n"

        return summary

    def ensure_backward_compatibility(self, meeting_id, agent_name):
        """Add backward compatibility for position data without confidence distributions."""
        if meeting_id not in self.positions or agent_name not in self.positions[meeting_id]:
            return

        for position in self.positions[meeting_id][agent_name]:
            # Add empty confidence distribution if not present
            if 'confidence_distribution' not in position:
                position['confidence_distribution'] = {}

                # Generate a simple confidence distribution based on position
                # This helps with backward compatibility visualization
                options = self._get_all_options_for_meeting(meeting_id)
                if options:
                    for option in options:
                        # Give high confidence to chosen option, distribute rest evenly
                        if option == position['option']:
                            position['confidence_distribution'][option] = 0.7  # 70% confidence
                        else:
                            position['confidence_distribution'][option] = 0.3 / (len(options) - 1) if len(
                                options) > 1 else 0

    def _get_all_options_for_meeting(self, meeting_id):
        """Get all unique options mentioned in a meeting."""
        if meeting_id not in self.positions:
            return []

        options = set()
        for agent_positions in self.positions[meeting_id].values():
            for position in agent_positions:
                # Add the chosen option
                if 'option' in position:
                    options.add(position['option'])
                # Add any options from confidence distribution
                if 'confidence_distribution' in position:
                    options.update(position['confidence_distribution'].keys())

        return list(options)

    def safely_get_position_data(self, meeting_id, agent_name, field='option', default=None):
        """Safely get position data with backward compatibility."""
        if not self._check_position_exists(meeting_id, agent_name):
            return default

        # Ensure backward compatibility
        self.ensure_backward_compatibility(meeting_id, agent_name)

        positions = self.positions[meeting_id][agent_name]
        if not positions:
            return default

        latest_position = positions[-1]
        return latest_position.get(field, default)

class VotingManager:
    """Manages voting records and statistics for all meetings in the simulation."""
    def __init__(self):
        """Initialize an empty voting manager."""
        # Main data structure: {meeting_id: {proposal_type: voting_data}}
        # Where proposal_type is either "initial" or "alternative"
        self.votes = {}  # Stores all voting data

    def initialize_meeting(self, meeting_id):
        """Initialize vote storage for a new meeting."""
        if meeting_id not in self.votes:
            self.votes[meeting_id] = {
                "initial": {
                    "proposal": None,
                    "proposal_speech": None,
                    "votes": {},
                    "reasons": {},
                    "passed": False,
                    "stats": {}
                },
                "alternative": {
                    "proposal": None,
                    "proposal_speech": None,
                    "votes": {},
                    "reasons": {},
                    "passed": False,
                    "stats": {}
                }
            }

    def record_votes(self, meeting_id, proposal, proposal_speech, votes, reasons, proposal_type="initial"):

        self.initialize_meeting(meeting_id)

        # Store proposal details
        self.votes[meeting_id][proposal_type]["proposal"] = proposal
        self.votes[meeting_id][proposal_type]["proposal_speech"] = proposal_speech
        self.votes[meeting_id][proposal_type]["votes"] = votes
        self.votes[meeting_id][proposal_type]["reasons"] = reasons

        # Calculate statistics
        total_votes = len(votes)
        approve_count = sum(1 for vote in votes.values() if vote == "Approve")
        approve_percentage = (approve_count / total_votes * 100) if total_votes > 0 else 0
        reject_count = total_votes - approve_count
        reject_percentage = 100 - approve_percentage

        # Check if passed (2/3 majority)
        passed = approve_count >= (2 / 3) * total_votes

        # Store results
        stats = {
            "total_votes": total_votes,
            "approve_count": approve_count,
            "approve_percentage": approve_percentage,
            "reject_count": reject_count,
            "reject_percentage": reject_percentage
        }

        self.votes[meeting_id][proposal_type]["stats"] = stats
        self.votes[meeting_id][proposal_type]["passed"] = passed

        return stats

    def get_vote_result(self, meeting_id, proposal_type="initial"):
        """Get the voting result (passed/failed) for a specific proposal."""
        if meeting_id in self.votes and proposal_type in self.votes[meeting_id]:
            return self.votes[meeting_id][proposal_type]["passed"]
        return False

    def get_vote_counts(self, meeting_id, proposal_type="initial"):
        """Get vote counts for a specific proposal."""
        if meeting_id in self.votes and proposal_type in self.votes[meeting_id]:
            return self.votes[meeting_id][proposal_type]["stats"]
        return None

    def get_shareholder_vote(self, meeting_id, shareholder_name, proposal_type="initial"):
        """Get a specific shareholder's vote."""
        if meeting_id in self.votes and proposal_type in self.votes[meeting_id]:
            return self.votes[meeting_id][proposal_type]["votes"].get(shareholder_name, None)
        return None

    def get_shareholder_reason(self, meeting_id, shareholder_name, proposal_type="initial"):
        """Get a specific shareholder's voting reason."""
        if meeting_id in self.votes and proposal_type in self.votes[meeting_id]:
            return self.votes[meeting_id][proposal_type]["reasons"].get(shareholder_name, None)
        return None

    def get_votes_by_type(self, meeting_id, vote_type, proposal_type="initial"):
        """Get all shareholders who voted a certain way (Approve/Reject)."""
        if meeting_id not in self.votes or proposal_type not in self.votes[meeting_id]:
            return []

        votes = self.votes[meeting_id][proposal_type]["votes"]
        return [name for name, vote in votes.items() if vote == vote_type]

    def get_all_votes_with_reasons(self, meeting_id, proposal_type="initial"):
        """Get all votes with their reasons."""
        if meeting_id not in self.votes or proposal_type not in self.votes[meeting_id]:
            return {}

        votes = self.votes[meeting_id][proposal_type]["votes"]
        reasons = self.votes[meeting_id][proposal_type]["reasons"]

        result = {}
        for shareholder, vote in votes.items():
            result[shareholder] = {
                "vote": vote,
                "reason": reasons.get(shareholder, "No reason provided")
            }

        return result

    def has_votes(self, meeting_id, proposal_type="initial"):
        """Check if votes exist for a meeting and proposal type."""
        return (meeting_id in self.votes and
                proposal_type in self.votes[meeting_id] and
                self.votes[meeting_id][proposal_type]["votes"])

    def get_summary_text(self, meeting_id, proposal_type="initial", include_reasons=True):
        """Generate a formatted summary of voting results.

        Args:
            meeting_id: Meeting identifier
            proposal_type: "initial" or "alternative"
            include_reasons: Whether to include voting reasons

        Returns:
            Formatted text summary of voting results
        """
        if not self.has_votes(meeting_id, proposal_type):
            return f"No voting data available for this {proposal_type} proposal."

        vote_data = self.votes[meeting_id][proposal_type]
        stats = vote_data["stats"]

        # Start with basic summary
        summary = f"## Voting Results: {proposal_type.capitalize()} Proposal\n\n"
        summary += f"Proposal: {vote_data['proposal']}\n\n"
        summary += f"Results: {stats['approve_count']}/{stats['total_votes']} votes in favor "
        summary += f"({stats['approve_percentage']:.1f}%). "
        summary += f"The proposal was {'APPROVED' if vote_data['passed'] else 'REJECTED'}.\n\n"

        # Add individual votes
        if include_reasons:
            summary += "Individual votes:\n"
            ceo_name = next((name for name in vote_data["votes"].keys()
                             if "ceo" in name.lower()), "CEO")

            # List CEO first
            if ceo_name in vote_data["votes"]:
                vote = vote_data["votes"][ceo_name]
                reason = vote_data["reasons"].get(ceo_name, "No reason provided")
                summary += f"- {ceo_name} (CEO): {vote} - {reason}\n"

            # Then list shareholders (non-CEO voters)
            for name, vote in vote_data["votes"].items():
                if name != ceo_name:  # Skip CEO as already listed
                    reason = vote_data["reasons"].get(name, "No reason provided")
                    summary += f"- {name}: {vote} - {reason}\n"

        return summary

    def get_ceo_decision_context(self, meeting_id, max_reasons=3):
        """Generate context specifically for CEO decision after failed vote.

        This provides a condensed format with just the information needed for
        the CEO to make a decision after a failed initial vote.
        """
        if not self.has_votes(meeting_id, "initial"):
            return "No voting data available."

        vote_data = self.votes[meeting_id]["initial"]
        if vote_data["passed"]:
            return "The proposal was approved, no decision needed."

        # Get stats
        stats = vote_data["stats"]

        # Create summary focused on rejection reasons
        context = f"Voting Results Summary:\n"
        context += f"Your proposal '{vote_data['proposal']}' was rejected with "
        context += f"{stats['approve_count']} approvals and {stats['reject_count']} rejections "
        context += f"({stats['approve_percentage']:.1f}% approval).\n\n"

        # Add rejection reasons (limited to max_reasons)
        context += "Key rejection reasons:\n"
        rejections = [(name, vote_data["reasons"].get(name, "No reason provided"))
                      for name, vote in vote_data["votes"].items()
                      if vote == "Reject" and name != self.get_ceo_name(vote_data["votes"])]

        # Take most important reasons (could be refined with importance ranking)
        selected_reasons = rejections[:max_reasons]

        for name, reason in selected_reasons:
            context += f"- {name}: {reason}\n"

        return context

    def get_ceo_name(self, votes_dict):
        """Helper to identify the CEO in a votes dictionary."""
        return next((name for name in votes_dict.keys()
                     if "ceo" in name.lower()), "CEO")

    def get_comprehensive_voting_history(self, meeting_id):
        """Get complete voting history for a meeting including both proposals."""
        if meeting_id not in self.votes:
            return "No voting records for this meeting."

        result = "# Complete Voting History\n\n"

        # Initial proposal
        if self.has_votes(meeting_id, "initial"):
            result += "## Initial Proposal\n\n"
            initial_data = self.votes[meeting_id]["initial"]
            result += f"Proposal: {initial_data['proposal']}\n"
            result += f"Result: {initial_data['stats']['approve_percentage']:.1f}% approval - "
            result += f"{'PASSED' if initial_data['passed'] else 'FAILED'}\n\n"

        # Alternative proposal (if it exists)
        if self.has_votes(meeting_id, "alternative"):
            result += "## Alternative Proposal\n\n"
            alt_data = self.votes[meeting_id]["alternative"]
            result += f"Proposal: {alt_data['proposal']}\n"
            result += f"Result: {alt_data['stats']['approve_percentage']:.1f}% approval - "
            result += f"{'PASSED' if alt_data['passed'] else 'FAILED'}\n\n"

        return result

    def get_shareholder_vote(self, meeting_id, shareholder_name, proposal_type="initial"):
        """Get a specific shareholder's vote."""
        if meeting_id in self.votes and proposal_type in self.votes[meeting_id]:
            return self.votes[meeting_id][proposal_type]["votes"].get(shareholder_name, None)
        return None

    def get_shareholder_reason(self, meeting_id, shareholder_name, proposal_type="initial"):
        """Get a specific shareholder's voting reason."""
        if meeting_id in self.votes and proposal_type in self.votes[meeting_id]:
            return self.votes[meeting_id][proposal_type]["reasons"].get(shareholder_name, None)
        return None

    def get_votes_by_type(self, meeting_id, vote_type, proposal_type="initial"):
        """Get all shareholders who voted a certain way (Approve/Reject)."""
        if meeting_id not in self.votes or proposal_type not in self.votes[meeting_id]:
            return []

        votes = self.votes[meeting_id][proposal_type]["votes"]
        return [name for name, vote in votes.items() if vote == vote_type]

    def get_all_votes_with_reasons(self, meeting_id, proposal_type="initial"):
        """Get all votes with their reasons."""
        if meeting_id not in self.votes or proposal_type not in self.votes[meeting_id]:
            return {}

        votes = self.votes[meeting_id][proposal_type]["votes"]
        reasons = self.votes[meeting_id][proposal_type]["reasons"]

        result = {}
        for shareholder, vote in votes.items():
            result[shareholder] = {
                "vote": vote,
                "reason": reasons.get(shareholder, "No reason provided")
            }

        return result

    def has_votes(self, meeting_id, proposal_type="initial"):
        """Check if votes exist for a meeting and proposal type."""
        return (meeting_id in self.votes and
                proposal_type in self.votes[meeting_id] and
                self.votes[meeting_id][proposal_type]["votes"])

    def get_ceo_name(self, votes_dict):
        """Helper to identify the CEO in a votes dictionary."""
        return next((name for name in votes_dict.keys()
                     if "ceo" in name.lower()), "CEO")

    def get_agent_vote_summary(self, meeting_id, agent_name, proposal_type="initial"):
        """Generate a personalized voting summary from an agent's perspective.
        Args:
            meeting_id: The meeting identifier
            agent_name: The specific agent's name
            proposal_type: "initial" or "alternative"
        Returns:
            Formatted text summary focused on the agent's own vote
        """
        if not self.has_votes(meeting_id, proposal_type):
            return f"No voting data available for this {proposal_type} proposal."

        vote_data = self.votes[meeting_id][proposal_type]
        proposal = vote_data["proposal"]

        # Get the agent's specific vote
        agent_vote = vote_data["votes"].get(agent_name, "Unknown")
        agent_reason = vote_data["reasons"].get(agent_name, "No reason provided")

        # Get overall outcome
        passed = vote_data["passed"]
        result_text = "APPROVED" if passed else "REJECTED"

        # Format the personalized summary
        summary = f"## My Vote on {proposal_type.capitalize()} Proposal\n\n"
        summary += f"Proposal: {proposal}\n\n"
        summary += f"My vote: {agent_vote}\n"
        summary += f"My reasoning: {agent_reason}\n\n"
        summary += f"Overall result: The proposal was {result_text}.\n"

        return summary


class DecisionTracker:
    """Tracks all decisions, discussions, and interactions throughout the simulation."""

    def __init__(self, ceo_name, ceo_mbti, shareholders):
        """Initialize the decision tracking structure."""
        self.data = {
            "CEO": {
                "name": ceo_name,
                "MBTI": ceo_mbti,
                "themes": {},
                "discussion_phase": {},
                "personal_summary": {},
                "personal_reflection": {},
                "personal_position": {},
                "decision_phase": {},
                "review_meeting": {}
            },
            "shareholders": {},
            "discussion_log": {},
            "company": {
                "CEO": {
                    "name": ceo_name,
                    "mbti": ceo_mbti
                },
                "shareholders": [],
                "company_assets": {
                    "initial_assets": 100000,
                    "final_assets": None,
                    "asset_change_history": [100000],
                    "quarterly_assets": {}
                },
                "company_decisions": {},
                "company_review": {}
            },
            "position": {
                "meetings": {}
            },
            "voting": {},
            "simulation": {
                "models": {
                    "model": None,
                    "model_low": None,
                    "model_high": None
                },
                "simulation_agents": len(shareholders) + 1,
                "simulation_years": None,
                "market_conditions": [],
                "random_seed": None,
                "asset_history": {
                    "assets": ["cash", "bonds", "real_estate", "stocks"],
                    "quarterly_returns": {}
                }
            }
        }

        # Initialize shareholders
        for shareholder_name, shareholder in shareholders.items():
            self.data["shareholders"][shareholder_name] = {
                "name": shareholder_name,
                "MBTI": shareholder.config.traits,
                "discussion_phase": {},
                "personal_summary": {},
                "personal_reflection": {},
                "personal_position": {},
                "decision_phase": {},
                "review_meeting": {}
            }

            self.data["company"]["shareholders"].append({
                "name": shareholder_name,
                "mbti": shareholder.config.traits
            })

        # Initialize empty position tracking
        self.data["position"]["meetings"] = {}
        self.data["voting"] = {}

    def record_simulation_metadata(self, years, random_seed, market_conditions, models):
        """Record simulation-level metadata."""
        self.data["simulation"]["simulation_years"] = years
        self.data["simulation"]["random_seed"] = random_seed
        self.data["simulation"]["market_conditions"] = market_conditions

        if models:
            self.data["simulation"]["models"]["model"] = str(models.get("main", ""))
            self.data["simulation"]["models"]["model_low"] = str(models.get("low", ""))
            self.data["simulation"]["models"]["model_high"] = str(models.get("high", ""))

    def record_market_returns(self, market_condition, quarterly_returns):
        """Record market returns for a specific condition."""
        if market_condition not in self.data["simulation"]["asset_history"]["quarterly_returns"]:
            self.data["simulation"]["asset_history"]["quarterly_returns"][market_condition] = {}

        for asset, returns in quarterly_returns.items():
            self.data["simulation"]["asset_history"]["quarterly_returns"][market_condition][asset] = returns

    def record_ceo_theme(self, meeting_id, initial_theme):
        """Record CEO's initial theme for a meeting."""
        if meeting_id not in self.data["CEO"]["themes"]:
            self.data["CEO"]["themes"][meeting_id] = {
                "initial_theme": initial_theme,
                "rounds": {}
            }

    def record_round_theme(self, meeting_id, round_num, theme_changed, reason, actual_theme):
        """Record theme changes for a specific round."""
        if meeting_id in self.data["CEO"]["themes"]:
            self.data["CEO"]["themes"][meeting_id]["rounds"][f"round_{round_num}"] = {
                "theme_changed": theme_changed,
                "reason": reason,
                "actual_theme": actual_theme
            }

    def record_ceo_opening_statement(self, meeting_id, opening_statement):
        """Record CEO's opening statement for a meeting."""
        if meeting_id not in self.data["CEO"]["discussion_phase"]:
            self.data["CEO"]["discussion_phase"][meeting_id] = {
                "opening_statement": opening_statement,
                "rounds": {}
            }
        else:
            self.data["CEO"]["discussion_phase"][meeting_id]["opening_statement"] = opening_statement

    def record_ceo_round_introduction(self, meeting_id, round_num, introduction):
        """Record CEO's theme introduction for a round."""
        if meeting_id in self.data["CEO"]["discussion_phase"]:
            if f"round_{round_num}" not in self.data["CEO"]["discussion_phase"][meeting_id]["rounds"]:
                self.data["CEO"]["discussion_phase"][meeting_id]["rounds"][f"round_{round_num}"] = {
                    "ceo_introduction": introduction,
                    "dialogue_content": {},
                    "ceo_summary": None,
                    "discussion_end": None
                }

    def record_shareholder_speech(self, meeting_id, round_num, shareholder_name, content, reason):
        """Record a shareholder's speech in a round."""
        # Record in CEO's section (existing functionality - keep as is)
        if meeting_id in self.data["CEO"]["discussion_phase"]:
            round_data = self.data["CEO"]["discussion_phase"][meeting_id]["rounds"].get(f"round_{round_num}")
            if round_data:
                if shareholder_name not in round_data["dialogue_content"]:
                    round_data["dialogue_content"][shareholder_name] = []

                round_data["dialogue_content"][shareholder_name].append({
                    "content": content,
                    "reason": reason,
                    "ceo_response": None,
                    "ceo_response_reason": None
                })

        # NEW: Record in the shareholder's own section with simplified structure
        if shareholder_name in self.data["shareholders"]:
            # Initialize shareholder's discussion_phase if needed
            if meeting_id not in self.data["shareholders"][shareholder_name]["discussion_phase"]:
                self.data["shareholders"][shareholder_name]["discussion_phase"][meeting_id] = {
                    "rounds": {}
                }

            # Initialize round if needed
            if f"round_{round_num}" not in self.data["shareholders"][shareholder_name]["discussion_phase"][meeting_id][
                "rounds"]:
                self.data["shareholders"][shareholder_name]["discussion_phase"][meeting_id]["rounds"][
                    f"round_{round_num}"] = {
                    "speeches": []
                }

            # Add the speech
            round_data = self.data["shareholders"][shareholder_name]["discussion_phase"][meeting_id]["rounds"][
                f"round_{round_num}"]
            round_data["speeches"].append({
                "content": content,
                "reason": reason
            })

    def record_ceo_response(self, meeting_id, round_num, shareholder_name, ceo_response, reason):
        """Record CEO's response to a shareholder."""
        if meeting_id in self.data["CEO"]["discussion_phase"]:
            round_data = self.data["CEO"]["discussion_phase"][meeting_id]["rounds"].get(f"round_{round_num}")
            if round_data and shareholder_name in round_data["dialogue_content"]:
                latest_entry = round_data["dialogue_content"][shareholder_name][-1]
                latest_entry["ceo_response"] = ceo_response
                latest_entry["ceo_response_reason"] = reason

    def record_ceo_round_summary(self, meeting_id, round_num, summary):
        """Record CEO's summary for a round."""
        if meeting_id in self.data["CEO"]["discussion_phase"]:
            round_data = self.data["CEO"]["discussion_phase"][meeting_id]["rounds"].get(f"round_{round_num}")
            if round_data:
                round_data["ceo_summary"] = summary

    def record_discussion_end_decision(self, meeting_id, round_num, should_end, reason):
        """Record CEO's decision to end discussion."""
        if meeting_id in self.data["CEO"]["discussion_phase"]:
            round_data = self.data["CEO"]["discussion_phase"][meeting_id]["rounds"].get(f"round_{round_num}")
            if round_data:
                round_data["discussion_end"] = {
                    "should_end": should_end,
                    "reason": reason
                }

    def record_personal_summary(self, agent_name, meeting_id, round_num, summary):
        """Record an agent's personal summary for a round."""
        if agent_name == self.data["CEO"]["name"]:
            entity = self.data["CEO"]
        elif agent_name in self.data["shareholders"]:
            entity = self.data["shareholders"][agent_name]
        else:
            return

        if meeting_id not in entity["personal_summary"]:
            entity["personal_summary"][meeting_id] = {}

        entity["personal_summary"][meeting_id][f"round_{round_num}"] = summary

    def record_personal_reflection(self, agent_name, meeting_id, round_num, reflection):
        """Record an agent's personal reflection for a round."""
        if agent_name == self.data["CEO"]["name"]:
            entity = self.data["CEO"]
        elif agent_name in self.data["shareholders"]:
            entity = self.data["shareholders"][agent_name]
        else:
            return

        if meeting_id not in entity["personal_reflection"]:
            entity["personal_reflection"][meeting_id] = {}

        entity["personal_reflection"][meeting_id][f"round_{round_num}"] = reflection

    def record_personal_position(self, agent_name, meeting_id, round_num, position_data):
        """Record an agent's position for a round."""
        if agent_name == self.data["CEO"]["name"]:
            entity = self.data["CEO"]
        elif agent_name in self.data["shareholders"]:
            entity = self.data["shareholders"][agent_name]
        else:
            return

        if meeting_id not in entity["personal_position"]:
            entity["personal_position"][meeting_id] = {
                "initial_position": None,
                "rounds": {}
            }

        if round_num == 0:
            entity["personal_position"][meeting_id]["initial_position"] = position_data
        else:
            entity["personal_position"][meeting_id]["rounds"][f"round_{round_num}"] = position_data

    def record_first_proposal(self, meeting_id, proposal, speech):
        """Record CEO's first proposal."""
        if meeting_id not in self.data["CEO"]["decision_phase"]:
            self.data["CEO"]["decision_phase"][meeting_id] = {
                "first_proposal": {"proposal": proposal, "speech": speech},
                "first_vote": {},
                "post_vote_decision": None,
                "second_proposal": None,
                "second_vote": {},
                "closing_statement": None,
                "executive_decision": None
            }
        else:
            self.data["CEO"]["decision_phase"][meeting_id]["first_proposal"] = {
                "proposal": proposal,
                "speech": speech
            }

    def record_agent_vote(self, meeting_id, agent_name, vote, reason, is_first_vote=True):
        """Record an agent's vote."""
        # Record in decision_phase
        if meeting_id in self.data["CEO"]["decision_phase"]:
            vote_key = "first_vote" if is_first_vote else "second_vote"

            if vote_key not in self.data["CEO"]["decision_phase"][meeting_id]:
                self.data["CEO"]["decision_phase"][meeting_id][vote_key] = {}

            if agent_name == self.data["CEO"]["name"]:
                self.data["CEO"]["decision_phase"][meeting_id][vote_key]["ceo_vote"] = vote
                self.data["CEO"]["decision_phase"][meeting_id][vote_key]["reason"] = reason
            else:
                if agent_name in self.data["shareholders"]:
                    if meeting_id not in self.data["shareholders"][agent_name]["decision_phase"]:
                        self.data["shareholders"][agent_name]["decision_phase"][meeting_id] = {}

                    self.data["shareholders"][agent_name]["decision_phase"][meeting_id][vote_key] = {
                        "vote": vote,
                        "reason": reason
                    }

        # Record in voting structure
        if meeting_id not in self.data["voting"]:
            self.data["voting"][meeting_id] = {
                "initial": None,
                "alternative": None
            }

        vote_type = "initial" if is_first_vote else "alternative"

        if self.data["voting"][meeting_id][vote_type] is None:
            self.data["voting"][meeting_id][vote_type] = {
                "proposal": None,
                "proposal_speech": None,
                "votes": {},
                "reasons": {},
                "passed": False,
                "stats": None
            }

        self.data["voting"][meeting_id][vote_type]["votes"][agent_name] = vote
        self.data["voting"][meeting_id][vote_type]["reasons"][agent_name] = reason

    def record_post_vote_decision(self, meeting_id, decision, reason):
        """Record CEO's decision after failed vote."""
        if meeting_id in self.data["CEO"]["decision_phase"]:
            self.data["CEO"]["decision_phase"][meeting_id]["post_vote_decision"] = {
                "decision": decision,
                "reason": reason
            }

    def record_second_proposal(self, meeting_id, proposal, speech):
        """Record CEO's second proposal."""
        if meeting_id in self.data["CEO"]["decision_phase"]:
            self.data["CEO"]["decision_phase"][meeting_id]["second_proposal"] = {
                "proposal": proposal,
                "speech": speech
            }

    def record_closing_statement(self, meeting_id, closing_statement):
        """Record CEO's closing statement."""
        if meeting_id in self.data["CEO"]["decision_phase"]:
            self.data["CEO"]["decision_phase"][meeting_id]["closing_statement"] = closing_statement

    def record_executive_decision(self, meeting_id, decision):
        """Record CEO's executive decision."""
        if meeting_id in self.data["CEO"]["decision_phase"]:
            self.data["CEO"]["decision_phase"][meeting_id]["executive_decision"] = decision

    def record_review_opening(self, meeting_id, opening_statement):
        """Record CEO's opening statement for review meeting."""
        if meeting_id not in self.data["CEO"]["review_meeting"]:
            self.data["CEO"]["review_meeting"][meeting_id] = {
                "opening_statement": opening_statement,
                "response_to_feedback": None,
                "closing_speech": None
            }
        else:
            self.data["CEO"]["review_meeting"][meeting_id]["opening_statement"] = opening_statement

    def record_review_response(self, meeting_id, response):
        """Record CEO's response to feedback."""
        if meeting_id in self.data["CEO"]["review_meeting"]:
            self.data["CEO"]["review_meeting"][meeting_id]["response_to_feedback"] = response

    def record_review_closing(self, meeting_id, closing_speech):
        """Record CEO's closing speech for review meeting."""
        if meeting_id in self.data["CEO"]["review_meeting"]:
            self.data["CEO"]["review_meeting"][meeting_id]["closing_speech"] = closing_speech

    def record_shareholder_review(self, meeting_id, shareholder_name, feedback, ratings):
        """Record shareholder's review feedback and ratings."""
        if shareholder_name in self.data["shareholders"]:
            if meeting_id not in self.data["shareholders"][shareholder_name]["review_meeting"]:
                self.data["shareholders"][shareholder_name]["review_meeting"][meeting_id] = {}

            self.data["shareholders"][shareholder_name]["review_meeting"][meeting_id] = {
                "written_feedback": feedback,
                "ratings": ratings
            }

    def record_company_assets(self, asset_amount, is_final=False):
        """Record company asset changes."""
        if is_final:
            self.data["company"]["company_assets"]["final_assets"] = asset_amount
        else:
            self.data["company"]["company_assets"]["asset_change_history"].append(asset_amount)

    def record_quarterly_investment(self, meeting_id, invested_amount, return_amount, return_rate):
        """Record quarterly investment results."""
        self.data["company"]["company_assets"]["quarterly_assets"][meeting_id] = {
            "invested_amount": invested_amount,
            "return_amount": return_amount,
            "return_rate": return_rate
        }

    def record_company_decision(self, meeting_id, final_choice, voting_details):
        """Record company-level decision for a meeting."""
        self.data["company"]["company_decisions"][meeting_id] = {
            "final_choice": final_choice,
            "voting_details": voting_details
        }

    def record_company_review(self, meeting_id, company_ratings, ceo_ratings):
        """Record company review results."""
        if meeting_id not in self.data["company"]["company_review"]:
            self.data["company"]["company_review"][meeting_id] = {}

        self.data["company"]["company_review"][meeting_id] = {
            "company_ratings": company_ratings,
            "ceo_ratings": ceo_ratings
        }

    def record_position_data(self, meeting_id, agent_name, round_num, option, reasoning, confidence_distribution,
                             changed):
        """Record position data in position structure."""
        if meeting_id not in self.data["position"]["meetings"]:
            self.data["position"]["meetings"][meeting_id] = {}

        if agent_name not in self.data["position"]["meetings"][meeting_id]:
            self.data["position"]["meetings"][meeting_id][agent_name] = []

        self.data["position"]["meetings"][meeting_id][agent_name].append({
            "round": round_num,
            "option": option,
            "reasoning": reasoning,
            "confidence_distribution": confidence_distribution,
            "changed": changed
        })

    def record_discussion_message(self, meeting_id, round_num, agent_name, message):
        """Record a message in the discussion log."""
        if meeting_id not in self.data["discussion_log"]:
            self.data["discussion_log"][meeting_id] = {}

        if f"round_{round_num}" not in self.data["discussion_log"][meeting_id]:
            self.data["discussion_log"][meeting_id][f"round_{round_num}"] = []

        self.data["discussion_log"][meeting_id][f"round_{round_num}"].append({
            "agent": agent_name,
            "message": message
        })

    def get_complete_data(self):
        """Get the complete tracked data structure."""
        return self.data

    def export_to_json(self, filepath):
        """Export all tracked data to a JSON file."""
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    def get_summary_statistics(self):
        """Generate summary statistics from tracked data."""
        summary = {
            "total_meetings": len(self.data["company"]["company_decisions"]),
            "total_rounds": 0,
            "total_votes": 0,
            "agent_participation": {},
            "decision_patterns": {},
            "performance_metrics": {
                "initial_assets": self.data["company"]["company_assets"]["initial_assets"],
                "final_assets": self.data["company"]["company_assets"]["final_assets"],
                "growth_rate": None
            }
        }

        # Calculate growth rate if final assets available
        if summary["performance_metrics"]["final_assets"]:
            initial = summary["performance_metrics"]["initial_assets"]
            final = summary["performance_metrics"]["final_assets"]
            summary["performance_metrics"]["growth_rate"] = ((final / initial) - 1) * 100

        # Count total rounds and agent participation
        for meeting_id, meeting_data in self.data["CEO"]["discussion_phase"].items():
            summary["total_rounds"] += len(meeting_data.get("rounds", {}))

            for round_key, round_data in meeting_data.get("rounds", {}).items():
                for shareholder_name, speeches in round_data.get("dialogue_content", {}).items():
                    if shareholder_name not in summary["agent_participation"]:
                        summary["agent_participation"][shareholder_name] = 0
                    summary["agent_participation"][shareholder_name] += len(speeches)

        # Count total votes
        for voting_data in self.data["voting"].values():
            for vote_type in ["initial", "alternative"]:
                if voting_data.get(vote_type) and voting_data[vote_type].get("votes"):
                    summary["total_votes"] += len(voting_data[vote_type]["votes"])

        return summary

    def get_complete_data(self):
        """Get the complete tracked data structure."""
        return self.data

    def export_to_json(self, filepath):
        """Export all tracked data to a JSON file."""
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    def get_summary_statistics(self):
        """Generate summary statistics from tracked data."""
        summary = {
            "total_meetings": len(self.data["company"]["company_decisions"]),
            "total_rounds": 0,
            "total_votes": 0,
            "agent_participation": {},
            "decision_patterns": {},
            "performance_metrics": {
                "initial_assets": self.data["company"]["company_assets"]["initial_assets"],
                "final_assets": self.data["company"]["company_assets"]["final_assets"],
                "growth_rate": None
            }
        }

        # Calculate growth rate if final assets available
        if summary["performance_metrics"]["final_assets"]:
            initial = summary["performance_metrics"]["initial_assets"]
            final = summary["performance_metrics"]["final_assets"]
            summary["performance_metrics"]["growth_rate"] = ((final / initial) - 1) * 100

        # Count total rounds and agent participation
        for meeting_id, meeting_data in self.data["CEO"]["discussion_phase"].items():
            summary["total_rounds"] += len(meeting_data.get("rounds", {}))

            for round_key, round_data in meeting_data.get("rounds", {}).items():
                for shareholder_name, speeches in round_data.get("dialogue_content", {}).items():
                    if shareholder_name not in summary["agent_participation"]:
                        summary["agent_participation"][shareholder_name] = 0
                    summary["agent_participation"][shareholder_name] += len(speeches)

        # Count total votes
        for voting_data in self.data["voting"].values():
            for vote_type in ["initial", "alternative"]:
                if voting_data.get(vote_type) and voting_data[vote_type].get("votes"):
                    summary["total_votes"] += len(voting_data[vote_type]["votes"])

        return summary

    def validate_data_integrity(self):
        """Validate the integrity of tracked data."""
        validation_results = {
            "valid": True,
            "issues": []
        }

        # Check CEO data
        if not self.data["CEO"]["name"]:
            validation_results["valid"] = False
            validation_results["issues"].append("CEO name is missing")

        # Check shareholders data
        for shareholder_name, shareholder_data in self.data["shareholders"].items():
            if not shareholder_data.get("MBTI"):
                validation_results["valid"] = False
                validation_results["issues"].append(f"MBTI missing for {shareholder_name}")

        # Check company assets
        if not self.data["company"]["company_assets"]["initial_assets"]:
            validation_results["valid"] = False
            validation_results["issues"].append("Initial assets not recorded")

        return validation_results

    def get_agent_summary(self, agent_name):
        """Get a comprehensive summary for a specific agent."""
        agent_summary = {
            "basic_info": {},
            "participation_stats": {},
            "position_changes": {},
            "voting_history": {},
            "review_ratings": {}
        }

        # Check if agent is CEO
        if agent_name == self.data["CEO"]["name"]:
            agent_summary["basic_info"] = {
                "role": "CEO",
                "mbti": self.data["CEO"]["MBTI"]
            }
            agent_summary["participation_stats"] = {
                "meetings_led": len(self.data["CEO"]["discussion_phase"]),
                "decisions_made": len(self.data["CEO"]["decision_phase"])
            }
        # Check if agent is shareholder
        elif agent_name in self.data["shareholders"]:
            agent_summary["basic_info"] = {
                "role": "Shareholder",
                "mbti": self.data["shareholders"][agent_name]["MBTI"]
            }

            # Count participation
            meetings_participated = 0
            total_speeches = 0
            for meeting_id, meeting_data in self.data["CEO"]["discussion_phase"].items():
                for round_key, round_data in meeting_data.get("rounds", {}).items():
                    if agent_name in round_data.get("dialogue_content", {}):
                        meetings_participated += 1
                        total_speeches += len(round_data["dialogue_content"][agent_name])

            agent_summary["participation_stats"] = {
                "meetings_participated": meetings_participated,
                "total_speeches": total_speeches
            }

        return agent_summary

    def get_meeting_timeline(self):
        """Generate a timeline of all meetings and decisions."""
        timeline = []

        # Process all company decisions chronologically
        for meeting_id, decision_data in self.data["company"]["company_decisions"].items():
            # Extract year and quarter from meeting_id
            year_match = re.search(r'year(\d+)', meeting_id)
            quarter_match = re.search(r'q(\d+)', meeting_id)

            year = int(year_match.group(1)) if year_match else 0
            quarter = int(quarter_match.group(1)) if quarter_match else 0

            timeline_entry = {
                "meeting_id": meeting_id,
                "year": year,
                "quarter": quarter,
                "decision": decision_data.get("final_choice"),
                "voting_result": decision_data.get("voting_details", {}).get("voting_result")
            }
            timeline.append(timeline_entry)

        # Sort by year and quarter
        timeline.sort(key=lambda x: (x["year"], x["quarter"]))

        return timeline


class MeetingManager:
    """Manages the mechanics of different meeting types (discussion flow, voting, etc.)"""

    # Class constants for discussion goals
    CEO_DISCUSSION_GOAL = """
CEO DISCUSSION GOAL:
Your primary responsibility is to facilitate identifying the optimal investment decision while demonstrating authentic leadership.

LEADERSHIP PRINCIPLES:
- EMBODY YOUR PERSONALITY TYPE: Let your MBTI personality type naturally influence your leadership style, information processing approach, and communication patterns throughout the discussion.
- FORM INITIAL PERSPECTIVE: Before discussion begins, develop your initial assessment of the options based on available information.
- SHARE YOUR THINKING: Clearly articulate your current perspective during your theme introduction, while expressing openness to refinement through collective wisdom.
- EXERCISE DISCERNMENT: Maintain a coherent assessment that evolves thoughtfully as valuable insights emerge from shareholder contributions.
- THOUGHTFULLY EVALUATE: Genuinely consider how others' perspectives might enhance your understanding - acknowledge valuable insights and integrate them into your evolving assessment.
- DEMONSTRATE LEADERSHIP: Balance conviction in your reasoning with authentic openness to new perspectives that improve collective understanding.
- FACILITATE OPTIMAL DECISION: Your ultimate goal is synthesizing diverse viewpoints to determine the option most beneficial for company development.

DISCUSSION MANAGEMENT:
- Focus on integrating valuable insights from all participants
- Limit discussion to analyzing information already available in the simulation
- Avoid complex financial metrics irrelevant to the core decision
- Respond substantively to shareholder perspectives, highlighting valuable elements
- Identify how each perspective contributes to developing a more complete understanding
- After each round, thoughtfully reconsider your position with genuine openness

PROFESSIONAL CONDUCT:
- Approach this as you would a real-world boardroom discussion with appropriate formality and professionalism
- Use natural business communication appropriate for executive-level discussions
- Make decisions with the seriousness and consideration a real investment would require
- Foster authentic collaborative discourse characteristic of high-functioning leadership teams

REMEMBER: Your goal is facilitating the optimal decision for company development. This requires synthesizing your own judgment with valuable shareholder insights, creating an environment where the best reasoning prevails regardless of its source.
"""

    SHAREHOLDER_DISCUSSION_GOAL = """
SHAREHOLDER DISCUSSION GOAL:
Your primary responsibility is to contribute meaningfully to identifying the optimal investment decision by offering your perspective and engaging thoughtfully with others' reasoning.

PARTICIPATION PRINCIPLES:
- EMBODY YOUR PERSONALITY TYPE: Let your MBTI personality type naturally influence which aspects of the discussion draw your attention and how you process information.
- FORM INITIAL PERSPECTIVE: Develop your initial assessment of options based on available information before discussion begins.
- CONTINUOUSLY RECALIBRATE: While maintaining a coherent perspective, genuinely integrate valuable insights that emerge during discussion.
- ENGAGE CONSTRUCTIVELY: Build upon and reference points from other participants, identifying how diverse perspectives might complement each other.
- THOUGHTFULLY EVALUATE: Consider how different viewpoints might enhance collective understanding, acknowledging valuable elements even in perspectives you initially disagree with.
- PROVIDE VALUABLE INSIGHT: Offer your unique perspective to help develop a more complete understanding of the decision context.
- FOCUS ON OPTIMAL OUTCOME: Your ultimate goal is helping identify the option that truly best serves company development.

DISCUSSION PARTICIPATION:
- Focus on how your contributions can enhance collective understanding
- Limit discussion to analyzing information already available in the simulation
- Avoid introducing complex financial metrics irrelevant to the decision
- Reference and build upon previous speakers' valuable points
- Express your reasoning clearly and concisely
- Remain genuinely open to refining your thinking when presented with compelling insights
- After each round, thoughtfully reconsider your position with authentic openness

PROFESSIONAL CONDUCT:
- Approach this as you would a real-world boardroom discussion with appropriate formality and professionalism
- Use natural business communication appropriate for investment discussions
- Present your views with the thoughtfulness real financial stakes would warrant
- Participate in authentic collaborative discourse characteristic of high-functioning teams

REMEMBER: Your goal is to help identify the optimal decision for company development by contributing your perspective and engaging meaningfully with the collective reasoning process.
"""

    def __init__(self, company, model):
        self.company = company
        self.model = model
        self.discussion_logs = {}  # Keep for compatibility
        self.market_condition_intros = {}
        # Add position manager as a class attribute
        self.position_manager = PositionManager()
        # Add the document manager
        self.document_manager = MeetingDocumentManager(model, company.embedder, company)
        # NEW: Add voting manager initialization
        self.voting_manager = VotingManager()

        # NEW: Add DecisionTracker reference
        if hasattr(company, 'decision_tracker'):
            self.decision_tracker = company.decision_tracker
        else:
            self.decision_tracker = None

    def run_public_discussion(self, meeting_id, meeting_type, options, market_condition, historical_returns=None,
                              budget=None):
        # Extract year and quarter from meeting_id
        year = int(meeting_id.split('_')[2].replace('year', ''))
        quarter = None
        if 'q' in meeting_id:
            quarter = int(meeting_id.split('_')[-1].replace('q', ''))

        # Initialize meeting documents - the return value is used for document management later
        if hasattr(self.document_manager, 'initialize_meeting'):
            self.document_manager.initialize_meeting(meeting_id, meeting_type, year, quarter)

        # Keep the discussion_log dict for compatibility
        discussion_log = {
            "meeting_id": meeting_id,
            "meeting_type": meeting_type,
            "market_condition": market_condition,
            "options": options,
            "rounds": []
        }

        # 1. CEO opening statement - add to all documents
        opening_statement = self._get_ceo_opening_statement(
            meeting_id, meeting_type, options, market_condition, historical_returns, budget
        )
        if hasattr(self.company, 'decision_tracker'):
            self.company.decision_tracker.record_ceo_opening_statement(meeting_id, opening_statement)
        discussion_log["opening_statement"] = opening_statement

        if hasattr(self.document_manager, 'add_to_all_documents'):
            self.document_manager.add_to_all_documents(
                f"## Meeting Opening\n\nCEO {self.company.ceo.config.name} opened the meeting with the following statement:\n\n{opening_statement}\n\n",
                tags=["opening_statement"]
            )

        # 1.A NEW: Initial position formation for all participants
        print(f"\n--- Forming initial positions for all participants ---")

        # CEO forms initial position - FIX HERE: Capture all three return values
        ceo_position, ceo_reasoning, ceo_confidence = self._form_initial_position_for_ceo(
            meeting_id, meeting_type, options, market_condition, historical_returns, budget
        )

        # Shareholders form initial positions in parallel
        shareholder_positions = self._parallel_form_initial_positions(
            meeting_id,
            meeting_type,
            options,
            market_condition,
            historical_returns,
            budget
        )
        # time.sleep(10)
        # 2. NEW: CEO creates discussion outline
        print(f"\n--- CEO creates discussion outline ---")
        discussion_outline = self._get_discussion_outline(
            meeting_id, meeting_type, ceo_position, ceo_reasoning, market_condition, historical_returns
        )

        discussion_log["ceo_internal"] = {
            "preferred_option": ceo_position,
            "discussion_outline": discussion_outline
        }

        # 3. Multiple rounds of discussion
        current_round = 0
        max_rounds = discussion_outline["num_rounds"]
        discussion_ended = False

        while not discussion_ended and current_round < max_rounds + 2:  # Add buffer for additional rounds
            current_round += 1

            # Add transition to new round
            print(f"\n--- Beginning Round {current_round} ---")

            round_transition = f"\n## Round {current_round} - Discussion\n\n"
            if hasattr(self.document_manager, 'add_to_all_documents'):
                self.document_manager.add_to_all_documents(round_transition,
                                                           tags=["transition", f"round_{current_round}",
                                                                 f"{meeting_id}_round{current_round}_filterable"])
            # Initialize round log
            round_log = {
                "round": current_round,
                "meeting_type": meeting_type,
                "meeting_id": meeting_id,
                "speeches": []
            }

            # 3.1 CEO determines the theme
            theme_decision = self._get_ceo_theme_decision(
                meeting_id, current_round, discussion_outline, discussion_log
            )
            round_log["theme"] = theme_decision["theme"]
            round_log["theme_decision"] = theme_decision

            theme_text = f"### Theme Decision\n\n"
            if theme_decision["changed"]:
                theme_text += f"CEO decided to adjust the theme for this round to: '{theme_decision['theme']}'\n"
                theme_text += f"Reason: {theme_decision['reason']}\n\n"
            else:
                theme_text += f"CEO decided to follow the original plan for this round with theme: '{theme_decision['theme']}'\n\n"

            if hasattr(self.document_manager, 'add_to_agent_only'):
                self.document_manager.add_to_agent_only(self.company.ceo.config.name, theme_text,
                                                        tags=[f"round_{current_round}_theme_decision"])

            # 3.2 CEO introduces the theme
            print(f"CEO introducing theme: {theme_decision['theme']}")

            ceo_intro = self._get_ceo_theme_introduction(
                meeting_id, current_round, theme_decision["theme"], discussion_log
            )
            round_log["speeches"].append({"name": self.company.ceo.config.name, "content": ceo_intro})

            intro_text = f"### Round {current_round} Introduction\n\n"
            intro_text += f"CEO {self.company.ceo.config.name}: \"{ceo_intro}\"\n\n"

            if hasattr(self.document_manager, 'add_to_all_documents'):
                self.document_manager.add_to_all_documents(intro_text,
                                                           tags=[f"round_{current_round}_introduction",
                                                                 f"{meeting_id}_round{current_round}_filterable"])

            # 3.3 Shareholders take turns having multi-round exchanges with the CEO
            discussion_text = f"### Discussion\n\n"
            if hasattr(self.document_manager, 'add_to_all_documents'):
                self.document_manager.add_to_all_documents(discussion_text,
                                                           tags=[f"round_{current_round}_discussion",
                                                                 f"{meeting_id}_round{current_round}_filterable"])
            shareholders = list(self.company.shareholders.values())
            random.shuffle(shareholders)  # Randomize speaking order

            for shareholder in shareholders:
                print(f"Shareholder {shareholder.config.name}'s turn")

                # Begin a multi-turn exchange between this shareholder and the CEO
                exchange_count = 0
                max_exchanges = 8  # Maximum number of back-and-forth exchanges
                exchange_continuing = True

                # Add a header for this shareholder's exchange
                shareholder_exchange_text = f"#### Exchange with {shareholder.config.name}\n\n"
                if hasattr(self.document_manager, 'add_to_all_documents'):
                    self.document_manager.add_to_all_documents(shareholder_exchange_text,
                                                               tags=[
                                                                   f"round_{current_round}_exchange_{shareholder.config.name}",
                                                                   f"{meeting_id}_round{current_round}_filterable"])
                # 3.3.1 Shareholder's initial statement - combined decision and content
                will_speak, speech_content = self._get_shareholder_speech(
                    meeting_id, current_round, shareholder, discussion_log, round_log
                )

                if will_speak:
                    # print(f"Shareholder {shareholder.config.name} speaks")
                    round_log["speeches"].append({"name": shareholder.config.name, "content": speech_content})
                    shareholder_text = f"{shareholder.config.name}: \"{speech_content}\"\n\n"
                    if hasattr(self.document_manager, 'add_to_all_documents'):
                        self.document_manager.add_to_all_documents(shareholder_text,
                                                                   tags=[f"round_{current_round}_speech",
                                                                         f"shareholder_{shareholder.config.name}",
                                                                         f"{meeting_id}_round{current_round}_filterable"])
                    exchange_count += 1
                else:
                    # print(f"Shareholder {shareholder.config.name} chooses not to speak")
                    round_log["speeches"].append({"name": shareholder.config.name, "content": "chooses not to speak"})
                    shareholder_text = f"{shareholder.config.name} chooses not to speak.\n\n"
                    if hasattr(self.document_manager, 'add_to_all_documents'):
                        self.document_manager.add_to_all_documents(shareholder_text,
                                                                   tags=[f"round_{current_round}_speech",
                                                                         f"shareholder_{shareholder.config.name}",
                                                                         f"{meeting_id}_round{current_round}_filterable"])
                    exchange_continuing = False

                # Continue the exchange as long as both parties are willing to continue
                while exchange_continuing and exchange_count < max_exchanges:
                    # If it's an even-numbered exchange, it's CEO's turn to respond
                    if exchange_count % 2 == 1:
                        will_respond, response_content = self._get_ceo_response_to_shareholder(
                            meeting_id, current_round, shareholder, discussion_log, round_log,
                            is_followup=(exchange_count > 1),  # Flag if this is a followup response
                            exchange_count=exchange_count  # Pass the current exchange count
                        )

                        if will_respond:
                            # print(f"CEO responds to {shareholder.config.name}")
                            round_log["speeches"].append(
                                {"name": self.company.ceo.config.name, "content": response_content})
                            response_text = f"CEO {self.company.ceo.config.name}: \"{response_content}\"\n\n"
                            if hasattr(self.document_manager, 'add_to_all_documents'):
                                self.document_manager.add_to_all_documents(response_text,
                                                                           tags=[f"round_{current_round}_response",
                                                                                 f"ceo_response",
                                                                                 f"{meeting_id}_round{current_round}_filterable"])
                            exchange_count += 1
                        else:
                            # print(f"CEO chooses not to respond to {shareholder.config.name}")
                            round_log["speeches"].append(
                                {"name": self.company.ceo.config.name, "content": "chooses not to respond"})
                            response_text = f"CEO {self.company.ceo.config.name} chooses not to respond.\n\n"
                            if hasattr(self.document_manager, 'add_to_all_documents'):
                                self.document_manager.add_to_all_documents(response_text,
                                                                           tags=[f"round_{current_round}_response",
                                                                                 f"ceo_response",
                                                                                 f"{meeting_id}_round{current_round}_filterable"])
                            exchange_continuing = False

                    # If it's an odd-numbered exchange, it's shareholder's turn to respond
                    else:
                        will_speak, speech_content = self._get_shareholder_followup_response(
                            meeting_id, current_round, shareholder, discussion_log, round_log,
                            exchange_count=exchange_count  # Pass the current exchange count
                        )

                        if will_speak:
                            # print(f"Shareholder {shareholder.config.name} responds")
                            round_log["speeches"].append({"name": shareholder.config.name, "content": speech_content})
                            shareholder_text = f"{shareholder.config.name}: \"{speech_content}\"\n\n"
                            if hasattr(self.document_manager, 'add_to_all_documents'):
                                self.document_manager.add_to_all_documents(shareholder_text,
                                                                           tags=[f"round_{current_round}_speech",
                                                                                 f"shareholder_{shareholder.config.name}",
                                                                                 f"{meeting_id}_round{current_round}_filterable"])
                            exchange_count += 1
                        else:
                            # print(f"Shareholder {shareholder.config.name} chooses not to respond")
                            round_log["speeches"].append(
                                {"name": shareholder.config.name, "content": "chooses not to respond"})
                            shareholder_text = f"{shareholder.config.name} chooses not to respond.\n\n"
                            if hasattr(self.document_manager, 'add_to_all_documents'):
                                self.document_manager.add_to_all_documents(shareholder_text,
                                                                           tags=[f"round_{current_round}_speech",
                                                                                 f"shareholder_{shareholder.config.name}",
                                                                                 f"{meeting_id}_round{current_round}_filterable"])
                            exchange_continuing = False

                # Add a visual separator between shareholder exchanges
                separator_text = "---\n\n"
                if hasattr(self.document_manager, 'add_to_all_documents'):
                    self.document_manager.add_to_all_documents(separator_text,
                                                               tags=[f"round_{current_round}_exchange_separator",
                                                                     f"{meeting_id}_round{current_round}_filterable"])
                # time.sleep(5)
            # 3.4 CEO summarizes the round
            print(f"CEO summarizing round {current_round}")

            summary_transition = f"### Round {current_round} Summary\n\n"
            if hasattr(self.document_manager, 'add_to_all_documents'):
                self.document_manager.add_to_all_documents(summary_transition,
                                                           tags=[f"round_{current_round}_summary_transition"])

            round_summary = self._get_ceo_round_summary(
                meeting_id, current_round, discussion_log, round_log
            )
            round_log["summary"] = round_summary

            summary_text = f"CEO {self.company.ceo.config.name} summarized: \"{round_summary}\"\n\n"
            if hasattr(self.document_manager, 'add_to_all_documents'):
                self.document_manager.add_to_all_documents(summary_text,
                                                           tags=[f"round_{current_round}_summary"])

            # 3.5 Update participants with round completion and trigger position reassessment
            self._update_participants_with_round(meeting_id, round_log, discussion_log)
            # time.sleep(5)
            # 3.6 CEO decides whether to end discussion
            print(f"CEO deciding whether to end discussion")

            end_transition = f"### End Round {current_round} Decision\n\n"
            if hasattr(self.document_manager, 'add_to_all_documents'):
                self.document_manager.add_to_all_documents(end_transition,
                                                           tags=[f"round_{current_round}_end_decision_transition"])

            end_decision, end_reason = self._get_ceo_end_decision(
                meeting_id, current_round, max_rounds, discussion_log, round_log
            )
            round_log[
                "end_decision"] = f"CEO decision: {'End' if end_decision else 'Continue'} discussion, reason: {end_reason}"
            # NEW: Record discussion end decision
            if hasattr(self.company, 'decision_tracker'):
                self.company.decision_tracker.record_discussion_end_decision(meeting_id, current_round, end_decision,
                                                                             end_reason)

            end_text = f"CEO {self.company.ceo.config.name} decided to {'end' if end_decision else 'continue'} the discussion.\n"
            end_text += f"Reason: {end_reason}\n\n"

            if hasattr(self.document_manager, 'add_to_all_documents'):
                self.document_manager.add_to_all_documents(end_text,
                                                           tags=[f"round_{current_round}_end_decision"])

            # Add completed round to discussion log
            discussion_log["rounds"].append(round_log)
            # NEW: Record round data in DecisionTracker
            if hasattr(self.company, 'decision_tracker'):
                # For each message in this round
                for speech in round_log.get("speeches", []):
                    if speech["name"] != "chooses not to speak" and speech["content"] != "chooses not to speak":
                        self.company.decision_tracker.record_discussion_message(
                            meeting_id=meeting_id,
                            round_num=current_round,
                            agent_name=speech["name"],
                            message=speech["content"]
                        )
            # Check if discussion should end
            discussion_ended = end_decision

            # *** NEW ADDITION: Now update the filter AFTER the CEO has made the decision ***
            if hasattr(self.document_manager, 'update_filter_for_round'):
                self.document_manager.update_filter_for_round(meeting_id, current_round)

            # Safety check - don't allow endless discussion
            if current_round >= max_rounds + 2:
                discussion_ended = True
                print("WARNING: Discussion exceeded maximum allowed rounds and was automatically ended.")

        # 4. Generate position tracking summary
        print(f"\n--- Generating position tracking summary ---")
        position_summary = self._track_position_changes(meeting_id)
        discussion_log["position_tracking"] = position_summary

        # 5. CEO selects final proposal
        print(f"\n--- CEO making final proposal ---")

        proposal_transition = f"\n## Final Proposal\n\n"
        if hasattr(self.document_manager, 'add_to_all_documents'):
            self.document_manager.add_to_all_documents(proposal_transition,
                                                       tags=["transition", "final_proposal"])

        final_proposal, proposal_reason = self._get_ceo_final_proposal(
            meeting_id, meeting_type, options, discussion_log
        )
        discussion_log["proposal"] = {
            "option": final_proposal,
            "reason": proposal_reason
        }

        proposal_text = f"CEO {self.company.ceo.config.name} made the following proposal:\n\n"
        proposal_text += f"\"Vote Proposal: {final_proposal}\n"
        proposal_text += f"Reasoning: {proposal_reason}\"\n\n"

        if hasattr(self.document_manager, 'add_to_all_documents'):
            self.document_manager.add_to_all_documents(proposal_text,
                                                       tags=["final_proposal_text", "meeting_log", "proposal_record",
                                                             meeting_id])
        # 6. Shareholders vote
        print(f"\n--- Shareholders voting on proposal ---")

        voting_transition = f"\n## Voting\n\n"
        if hasattr(self.document_manager, 'add_to_all_documents'):
            self.document_manager.add_to_all_documents(voting_transition,
                                                       tags=["transition", "voting"])

        # MODIFIED: Use VotingManager for vote collection
        votes, vote_reasons, passed = self._collect_shareholder_votes(
            meeting_id, meeting_type, final_proposal, proposal_reason, discussion_log
        )
        discussion_log["votes"] = votes
        discussion_log["vote_reasons"] = vote_reasons
        discussion_log["passed"] = passed

        # 7. Handle voting result
        print(f"\n--- Processing voting results (passed: {passed}) ---")

        result_transition = f"\n## Meeting Conclusion\n\n"
        if hasattr(self.document_manager, 'add_to_all_documents'):
            self.document_manager.add_to_all_documents(result_transition,
                                                       tags=["transition", "conclusion"])

        # MODIFIED: Handle voting result using VotingManager
        if passed:
            # Proposal passed, CEO makes closing statement
            # MODIFIED: Updated function call to remove unused parameters
            closing_statement = self._get_ceo_closing_statement_after_vote(
                meeting_id, final_proposal
            )
            discussion_log["closing_statement"] = closing_statement

            closing_text = f"CEO {self.company.ceo.config.name} made the following closing statement:\n\n"
            closing_text += f"\"{closing_statement}\"\n\n"
            closing_text += f"The meeting concluded with {final_proposal} as the approved decision.\n\n"

            if hasattr(self.document_manager, 'add_to_all_documents'):
                self.document_manager.add_to_all_documents(closing_text, tags=["closing_statement"])
            selected_option = final_proposal

        else:
            # Proposal failed, get CEO's decision: force through or modify
            print(f"\n--- Handling failed vote ---")

            failed_vote_transition = f"\n## Response to Failed Vote\n\n"
            if hasattr(self.document_manager, 'add_to_all_documents'):
                self.document_manager.add_to_all_documents(failed_vote_transition,
                                                           tags=["transition", "failed_vote_response"])

            # MODIFIED: Updated function call to remove unused parameters
            ceo_decision, decision_reason = self._get_ceo_decision_after_failed_vote(
                meeting_id, meeting_type, final_proposal
            )

            decision_text = f"After the proposal was rejected, CEO {self.company.ceo.config.name} decided to {ceo_decision.lower()}.\n"
            decision_text += f"Reason: {decision_reason}\n\n"

            if hasattr(self.document_manager, 'add_to_all_documents'):
                self.document_manager.add_to_all_documents(decision_text, tags=["ceo_post_vote_decision"])

            discussion_log["post_vote_decision"] = {
                "decision": ceo_decision,
                "reason": decision_reason
            }

            if ceo_decision == "Force through original proposal":
                # CEO decides to implement the original proposal despite objections
                print(f"\n--- CEO forcing through original proposal ---")

                force_through_text = f"CEO {self.company.ceo.config.name} announced: \"Despite the vote results, I have decided to proceed with {final_proposal}. {decision_reason}\"\n\n"

                if hasattr(self.document_manager, 'add_to_all_documents'):
                    self.document_manager.add_to_all_documents(force_through_text, tags=["force_through_announcement"])

                # MODIFIED: Updated function call to remove unused parameters
                closing_statement = self._get_ceo_closing_statement_after_force_through(
                    meeting_id, final_proposal
                )
                discussion_log["closing_statement"] = closing_statement

                closing_text = f"CEO {self.company.ceo.config.name} made the following closing statement:\n\n"
                closing_text += f"\"{closing_statement}\"\n\n"
                closing_text += f"The meeting concluded with {final_proposal} as the final decision (enforced by CEO).\n\n"

                if hasattr(self.document_manager, 'add_to_all_documents'):
                    self.document_manager.add_to_all_documents(closing_text, tags=["closing_statement"])
                selected_option = final_proposal

            else:
                # CEO decides to create a modified proposal
                print(f"\n--- CEO offering alternative proposal ---")

                alternative_transition = f"\n## Alternative Proposal\n\n"
                if hasattr(self.document_manager, 'add_to_all_documents'):
                    self.document_manager.add_to_all_documents(alternative_transition,
                                                               tags=["transition", "alternative_proposal"])

                # MODIFIED: Updated function call to remove unused parameters
                alternative_proposal, alternative_reason = self._get_ceo_alternative_proposal(
                    meeting_id, meeting_type, options, discussion_log
                )

                # Record CEO's alternative proposal
                alt_proposal_text = f"CEO {self.company.ceo.config.name} made an alternative proposal:\n\n"
                alt_proposal_text += f"\"Since the previous proposal was not approved, I now propose {alternative_proposal}. {alternative_reason}\"\n\n"

                if hasattr(self.document_manager, 'add_to_all_documents'):
                    self.document_manager.add_to_all_documents(alt_proposal_text, tags=["alternative_proposal_text"])

                # Add second round voting transition
                alt_voting_transition = f"\n## Alternative Proposal Voting\n\n"
                if hasattr(self.document_manager, 'add_to_all_documents'):
                    self.document_manager.add_to_all_documents(alt_voting_transition,
                                                               tags=["transition", "alternative_voting"])

                # Save alternative proposal info to discussion_log
                discussion_log["alternative_proposal"] = {
                    "option": alternative_proposal,
                    "reason": alternative_reason
                }

                # Second round of voting
                print(f"\n--- Voting on alternative proposal ---")

                alternative_votes, alternative_vote_reasons, alternative_passed = self._collect_shareholder_votes(
                    meeting_id, meeting_type, alternative_proposal, alternative_reason, discussion_log,
                    is_alternative_proposal=True  # Added parameter for VotingManager
                )
                # Save voting results to discussion_log for backward compatibility
                discussion_log["alternative_votes"] = alternative_votes
                discussion_log["alternative_vote_reasons"] = alternative_vote_reasons
                discussion_log["alternative_passed"] = alternative_passed

                # MODIFIED: Get voting results display from VotingManager
                alt_vote_text = self.voting_manager.get_summary_text(meeting_id, "alternative")
                if hasattr(self.document_manager, 'add_to_all_documents'):
                    self.document_manager.add_to_all_documents(alt_vote_text, tags=["alternative_voting_results"])

                # CEO closing statement
                # MODIFIED: Updated function call to remove unused parameters
                closing_statement = self._get_ceo_closing_statement_after_vote(
                    meeting_id, alternative_proposal
                )
                discussion_log["closing_statement"] = closing_statement

                closing_text = f"CEO {self.company.ceo.config.name} made the following closing statement:\n\n"
                closing_text += f"\"{closing_statement}\"\n\n"

                if hasattr(self.document_manager, 'add_to_all_documents'):
                    self.document_manager.add_to_all_documents(closing_text, tags=["closing_statement"])

                if alternative_passed:
                    # Proposal passed - get closing statement
                    closing_statement = self._get_ceo_closing_statement_after_vote(
                        meeting_id, alternative_proposal
                    )
                    discussion_log["closing_statement"] = closing_statement

                    closing_text = f"CEO {self.company.ceo.config.name} made the following closing statement:\n\n"
                    closing_text += f"\"{closing_statement}\"\n\n"

                    if hasattr(self.document_manager, 'add_to_all_documents'):
                        self.document_manager.add_to_all_documents(closing_text, tags=["closing_statement"])

                    selected_option = alternative_proposal
                    conclusion_text = f"The meeting concluded with {alternative_proposal} as the approved decision.\n\n"

                    if hasattr(self.document_manager, 'add_to_all_documents'):
                        self.document_manager.add_to_all_documents(conclusion_text, tags=["meeting_conclusion"])
                else:
                    # If even alternative fails, CEO makes executive decision
                    print(f"\n--- CEO making executive decision ---")

                    executive_transition = f"\n## Executive Decision\n\n"
                    if hasattr(self.document_manager, 'add_to_all_documents'):
                        self.document_manager.add_to_all_documents(executive_transition,
                                                                   tags=["transition", "executive_decision"])

                    # MODIFIED: Updated function call to remove unused parameters
                    selected_option, executive_reason = self._get_ceo_executive_decision(meeting_id, meeting_type, options)
                    discussion_log["executive_decision"] = selected_option

                    executive_text = f"As neither proposal was approved, CEO {self.company.ceo.config.name} made an executive decision:\n\n"
                    executive_text += f"\"As we could not reach consensus, as CEO I have decided to choose {selected_option}. {executive_reason}\"\n\n"

                    executive_text += f"The meeting concluded with {selected_option} as the final decision.\n\n"

                    if hasattr(self.document_manager, 'add_to_all_documents'):
                        self.document_manager.add_to_all_documents(executive_text, tags=["executive_decision"])

        # Store the meeting log for backward compatibility
        self.discussion_logs[meeting_id] = discussion_log

        # # Finalize meeting documents
        # if hasattr(self.document_manager, 'finalize_meeting'):
        #     self.document_manager.finalize_meeting()

        return selected_option, discussion_log

    def run_written_submission(self, meeting_id, meeting_type, year_performance):
        # Extract year from meeting_id
        year = int(meeting_id.split('_')[2].replace('year', ''))

        # Initialize meeting documents - call for side effects but don't store return value
        self.document_manager.initialize_meeting(meeting_id, meeting_type, year)

        # Get or create evaluation criteria
        evaluation_criteria = self._define_evaluation_criteria()

        # Create meeting context dictionary
        meeting_log = {
            "meeting_id": meeting_id,
            "meeting_type": meeting_type,
            "year_performance": year_performance
        }

        # Get meeting summaries
        meeting_summaries = self._create_meeting_summaries_for_review(year)
        meeting_log["meeting_summaries"] = meeting_summaries

        # CEO opening statement summarizing the year
        opening_transition = "## Annual Review Meeting\n\n"
        self.document_manager.add_to_all_documents(opening_transition, tags=["transition", "meeting_opening"])

        ceo_statement = self._get_ceo_opening_statement_for_review(
            meeting_id, year_performance, evaluation_criteria
        )
        if hasattr(self.company, 'decision_tracker'):
            self.company.decision_tracker.record_review_opening(meeting_id, ceo_statement)
        meeting_log["ceo_statement"] = ceo_statement

        statement_text = f"CEO {self.company.ceo.config.name} opened the meeting with the following statement:\n\n"
        statement_text += f"\"{ceo_statement}\"\n\n"
        self.document_manager.add_to_all_documents(statement_text, tags=["ceo_opening_statement"])

        # Collect written submissions and questions from shareholders
        submissions_transition = "\n## Shareholder Written Evaluations\n\n"
        self.document_manager.add_to_all_documents(submissions_transition,
                                                   tags=["transition", "shareholder_submissions"])

        shareholder_data = self._collect_shareholder_submissions(
            meeting_id, ceo_statement
        )
        submissions = shareholder_data.get("submissions", {})
        questions = shareholder_data.get("questions", {})

        meeting_log["shareholder_submissions"] = submissions
        meeting_log["shareholder_questions"] = questions

        # Add submissions to document
        for shareholder_name, submission in submissions.items():
            submission_text = f"### {shareholder_name}'s Evaluation\n\n{submission}\n\n"

            # Add to submitting shareholder and CEO only
            self.document_manager.add_to_specific_documents(
                [shareholder_name, self.company.ceo.config.name],
                submission_text,
                tags=["shareholder_submission", f"submission_{shareholder_name}"]
            )

        # CEO response to submissions and questions
        response_transition = "\n## CEO Response to Evaluations\n\n"
        self.document_manager.add_to_all_documents(response_transition,
                                                   tags=["transition", "ceo_response"])

        ceo_response = self._get_ceo_response_to_submissions(
            meeting_id, shareholder_data
        )
        meeting_log["ceo_response"] = ceo_response

        response_text = f"CEO {self.company.ceo.config.name} responded to the evaluations and questions:\n\n"
        response_text += f"\"{ceo_response}\"\n\n"
        self.document_manager.add_to_all_documents(response_text, tags=["annual_review", f"year_{year}","ceo_response_text"])

        # Collect multi-dimensional ratings
        ratings_transition = "\n## Performance Ratings\n\n"
        self.document_manager.add_to_all_documents(ratings_transition,
                                                   tags=["transition", "performance_ratings"])

        ratings_data = self._collect_shareholder_ratings(meeting_id)
        meeting_log["ratings"] = ratings_data

        # Get average ratings for the core dimensions
        avg_ceo_rating = ratings_data["averages"]["ceo"]["overall"]
        avg_company_rating = ratings_data["averages"]["company"]["overall"]

        # CEO closing statement
        closing_transition = "\n## Meeting Conclusion\n\n"
        self.document_manager.add_to_all_documents(closing_transition,
                                                   tags=["transition", "meeting_conclusion"])

        ceo_closing = self._get_ceo_closing_statement_for_review(
            meeting_id, ratings_data
        )
        meeting_log["ceo_closing"] = ceo_closing

        closing_text = f"CEO {self.company.ceo.config.name} concluded the meeting:\n\n"
        closing_text += f"\"{ceo_closing}\"\n\n"
        closing_text += "The annual review meeting has concluded.\n\n"

        self.document_manager.add_to_all_documents(closing_text, tags=["meeting_conclusion_text"])

        # Finalize meeting documents and get summaries
        summaries = self.document_manager.finalize_meeting()

        # Add summaries to meeting_log for future reference
        meeting_log["document_summaries"] = summaries

        # Return legacy structure for backward compatibility plus detailed ratings
        return {
            "ceo_rating": avg_ceo_rating,
            "company_rating": avg_company_rating,
            "detailed_ratings": ratings_data
        }, meeting_log

    def generate_meeting_id(self, meeting_type, year, quarter=None):
        """Generate a unique ID for each meeting for tracking purposes."""
        if quarter:
            return f"{meeting_type}_year{year}_q{quarter}"
        return f"{meeting_type}_year{year}"

    def _get_ceo_opening_statement(self, meeting_id, meeting_type, options=None, market_condition=None,
                                   historical_returns=None, budget=None):
        """Helps CEO craft an opening statement for meetings."""

        # Extract year and quarter information from meeting_id
        year = int(meeting_id.split('_')[2].replace('year', ''))
        quarter = None
        if 'q' in meeting_id:
            quarter = int(meeting_id.split('_')[-1].replace('q', ''))

        # Check if historical data is available
        has_historical_data = (year > 1) or (quarter and quarter > 1) or (
                historical_returns and any(len(returns) > 0 for asset, returns in historical_returns.items()))

        # Get current assets
        company_assets = self.company.assets

        # Get market condition introduction
        market_intro = self.market_condition_intros.get(market_condition, "")

        # Create opening statement based on meeting type
        if "annual_budget" in meeting_type:
            # Annual Budget Meeting
            welcome = f"Welcome everyone to our annual budget meeting for Year {year}. Thank you all for joining today."

            # Add market intro
            status = f"Our company currently has ${company_assets:.2f} in total assets. We are operating in a {market_condition} market environment this year. {market_intro}"

            # Get CEO's analysis of previous year (if not first year)
            previous_year_analysis = ""
            if year > 1:
                # Get previous year performance data
                prev_year_performance = None
                if hasattr(self.company, 'meeting_logs'):
                    prev_logs = [log for log in self.company.meeting_logs
                                 if 'annual_review' in log.get('meeting_id', '') and f'year{year - 1}' in log.get(
                            'meeting_id', '')]
                    if prev_logs:
                        prev_year_performance = prev_logs[0].get('year_performance', None)

                if prev_year_performance:
                    escaped_performance = escape_curly_braces(str(prev_year_performance))
                    analysis_prompt = entity.free_action_spec(
                        call_to_action=f"As {self.company.ceo.config.name}, provide a 2-3 sentence analysis of the previous year's performance. Previous year data: {escaped_performance}. Consider your MBTI personality type when crafting this response.",
                        tag=f"annual_budget_year{year}_analysis"
                    )
                    previous_year_analysis = f"\n\nRegarding our previous year's performance: {self.company.ceo.agent.act(analysis_prompt)}"

            # Add historical returns data for all previous years
            historical_data_presentation = ""
            if has_historical_data:
                if year > 1:
                    # Calculate the quarter index for the end of previous year
                    up_to_quarter = (year - 1) * 4
                    # Get formatted historical returns for all previous years
                    historical_data_presentation = f"\n\n{self.company.market.get_formatted_historical_returns(up_to_quarter)}"
            else:
                # In case of no historical data
                historical_data_presentation = "\n\nAs this is our first year, we do not have historical performance data yet. We'll need to base our decisions on theoretical risk-return profiles and current market conditions."

            # Add investment history presentation
            investment_history_presentation = ""
            if year > 1:  # Only add if not the first year
                investment_history_presentation = f"\n\n{self.company.get_formatted_investment_history(year, 1)}"

            # Add budget history presentation for annual budget meetings (Year 2+)
            budget_history_presentation = ""
            if year > 1:
                budget_history_presentation = f"\n\n{self.company.get_formatted_budget_history(year)}"

            # Fixed budget options explanation
            options_explanation = """
    The today, we need to decide how we'll allocate our investment budget across the four quarters of this year. We have four options to consider:

    - Option A: We allocate 50% of our total funds, evenly distributed across the four quarters (12.5% per quarter).
    - Option B: We allocate 50% of our total funds with an increasing allocation: 5% in Q1, 10% in Q2, 15% in Q3, and 20% in Q4.
    - Option C: We allocate 100% of our total funds, evenly distributed across the four quarters (25% per quarter).
    - Option D: We allocate 100% of our total funds with an increasing allocation: 10% in Q1, 20% in Q2, 30% in Q3, and 40% in Q4.

    Each option presents different levels of risk and opportunity given the current market conditions."""

            objectives = f"""
    The objective of today's meeting is to thoroughly discuss these options and come to a decision that best serves our company's interests in this {market_condition} market environment. We'll need to balance potential returns with the associated risks. We'll have an open discussion, then I'll propose what I believe is the best approach based on our collective input, and we'll take a vote requiring a two-thirds majority to proceed.

    Let's begin our discussion with your initial thoughts on these options."""

            # Combine all sections
            opening_statement = welcome + "\n\n" + status + previous_year_analysis + historical_data_presentation + investment_history_presentation + budget_history_presentation + "\n" + options_explanation + "\n" + objectives

        elif "quarterly_investment" in meeting_type:
            # Quarterly Investment Meeting
            welcome = f"Welcome to our Q{quarter} investment meeting for Year {year}. I appreciate everyone's presence and input today."

            status = f"For this quarter, we have ${budget:.2f} allocated for investment in this {market_condition} market environment. {market_intro}"

            # Add historical returns data presentation
            historical_data_presentation = ""
            if has_historical_data:
                up_to_quarter = (year - 1) * 4 + quarter - 1
                if up_to_quarter > 0:
                    historical_data_presentation = f"\n\n{self.company.market.get_formatted_historical_returns(up_to_quarter)}"
            else:
                # In case of no historical data
                if quarter == 1 and year == 1:
                    historical_data_presentation = "\n\nAs this is our first investment decision, we don't have historical performance data yet. We'll need to base our decisions on theoretical risk-return profiles and current market conditions."
                elif quarter > 1:
                    # If it's a later quarter in the first year, we might have some limited data
                    up_to_quarter = quarter - 1
                    if up_to_quarter > 0:
                        historical_data_presentation = f"\n\n{self.company.market.get_formatted_historical_returns(up_to_quarter)}"

            # Add investment history presentation
            investment_history_presentation = ""
            if quarter > 1 or year > 1:  # Only add if not the first quarter of the first year
                investment_history_presentation = f"\n\n{self.company.get_formatted_investment_history(year, quarter)}"

            # Get CEO's analysis of historical returns
            historical_analysis = ""
            if has_historical_data and historical_returns and any(
                    historical_returns[asset] for asset in historical_returns):
                # Safely format historical_returns
                safe_historical_returns = escape_curly_braces(str(historical_returns))
                analysis_prompt = entity.free_action_spec(
                    call_to_action=f"As {self.company.ceo.config.name}, provide a brief 2-3 sentence analysis of the historical returns of our investment options. Historical data: {safe_historical_returns}. Focus on patterns or trends relevant to our current {market_condition} market. Consider both returns and risks. Consider your MBTI personality type.",
                    tag=f"quarterly_investment_year{year}_q{quarter}_analysis"
                )
                historical_analysis = f"\n\nMy analysis of our historical performance: {self.company.ceo.agent.act(analysis_prompt)}"

            # Fixed investment options explanation
            options_explanation = """
    Today, we need to decide which asset to invest in for this quarter. Our options are:

    - Cash Savings: Low risk, low return.
    - Bonds: Low to medium risk, low to medium return.
    - Real Estate: Medium risk, medium return.
    - Stocks: High risk, high return.

    Each asset class has different risk-return profiles that may perform differently given our current market conditions."""

            objectives = f"""
    Our objective today is to thoroughly evaluate these options, consider our company's risk tolerance in the current {market_condition} market, and select the investment that offers the best balance of risk and potential return. We'll have an open discussion, after which I'll propose what I believe is the optimal asset for this quarter, followed by a vote requiring a two-thirds majority to proceed.

    Let's begin by sharing your thoughts on the current market environment and which asset might be most appropriate."""

            # Combine all sections
            opening_statement = welcome + "\n\n" + status + historical_data_presentation + investment_history_presentation + historical_analysis + "\n" + options_explanation + "\n" + objectives

        else:
            # Default case - should not happen but providing fallback
            welcome = f"Welcome to our meeting for Year {year}."
            # Simplified opening for unknown meeting types
            opening_statement = welcome + f"\n\nWe are gathered to discuss our company's operations in this {market_condition} market environment."

        return opening_statement

    def _form_initial_position_for_ceo(self, meeting_id, meeting_type, options, market_condition,
                                       historical_returns=None, budget=None):
        """Helps CEO decide their initial position with confidence distribution."""
        # Extract year and quarter from meeting_id
        year = int(meeting_id.split('_')[2].replace('year', ''))
        quarter = None
        if 'q' in meeting_id:
            quarter = int(meeting_id.split('_')[-1].replace('q', ''))

        # Check if historical data is available
        has_historical_data = (year > 1) or (quarter and quarter > 1) or (
                historical_returns and any(len(returns) > 0 for asset, returns in historical_returns.items()))

        # Create options string for the original format
        options_str = "\n".join([f"{k}: {v}" for k, v in options.items()])

        # Create options string for confidence distribution format
        options_conf_str = "\n".join([f"- {k}: [confidence percentage]" for k in options.keys()])

        # Get the CEO's context for decision making
        ceo_context = ""
        if hasattr(self.document_manager, 'get_agent_context'):
            ceo_context = self.document_manager.get_agent_context(self.company.ceo.config.name)

        # Different prompts based on meeting type
        if "annual_budget" in meeting_type:
            # Add market introduction
            market_intro = self.market_condition_intros.get(market_condition, "")

            # Create historical performance summary if available
            historical_summary = ""
            if has_historical_data:
                if year > 1:
                    # With historical data
                    historical_summary = "\nHistorical performance data from previous years is available.\n"
                    historical_summary += "Consider how each budget allocation strategy performed in the past.\n"
            else:
                # No historical data
                historical_summary = "\nAs this is our first year, we do not have historical performance data yet.\n"
                historical_summary += "Base your decision on theoretical understanding of the current market conditions.\n"

            preference_prompt = f"""You are {self.company.ceo.config.name}, the CEO of the investment company with MBTI personality type {self.company.ceo.config.traits}.

Before beginning the annual budget meeting discussion, you need to form your initial position on which budget option you believe is best. This is your private assessment that will guide your stance throughout the discussion.

{self.CEO_DISCUSSION_GOAL}

Current situation:
- You're planning for the annual budget meeting in a {market_condition} market environment
- Market details: {market_intro}
- Company current assets: ${self.company.assets:.2f}
- Available budget options:
{options_str}
{historical_summary}

Based on your MBTI personality type ({self.company.ceo.config.traits}), the current market conditions, and available data, assess your confidence in each budget allocation option and select which you personally believe would be best for the company this year.
IMPORTANT: Frame your thinking in first person ("I/my") as an internal reflection process. Use step-by-step analysis (think through each element of your decision sequentially) to demonstrate your reasoning process. Express your analytical considerations and personal assessment with authentic introspection.
IMPORTANT: This initial position will guide your leadership throughout the discussion. While you remain open to changing your mind if compelling arguments arise, you should advocate for this position and not abandon it easily.

Your response MUST follow this exact format:
CONFIDENCE DISTRIBUTION:
{options_conf_str}
PREFERRED OPTION: [state your preferred option - the one with highest confidence]
REASONING: [explain your reasoning in 3-5 sentences that reflect your MBTI personality type]

Example format for annual budget meeting:
CONFIDENCE DISTRIBUTION:
- Option A: 15%
- Option B: 5%
- Option C: 75%
- Option D: 5%
PREFERRED OPTION: Option C
REASONING: As a strategic leader in this stable market environment, I believe full allocation (100%) with even distribution provides us the most consistent growth opportunity. This approach balances risk across all quarters while maximizing our capital utilization, which aligns with my analytical yet decisive approach to financial management.

IMPORTANT FORMAT INSTRUCTION: Your response MUST strictly follow the exact output format specified above. Do not add any explanations, notes, or content outside this format. Any deviation from the required format structure will cause processing errors. Provide ONLY the formatted response as outlined.

===MEETING CONTEXT===
    {ceo_context}
===MEETING CONTEXT==="""
        else:  # quarterly investment
            # Extract year and quarter from meeting_id
            year = int(meeting_id.split('_')[2].replace('year', ''))
            quarter = int(meeting_id.split('_')[-1].replace('q', ''))

            # Create historical performance summary if available
            historical_summary = ""
            if has_historical_data and historical_returns:
                assets = list(historical_returns.keys())
                quarters = len(historical_returns[assets[0]])
                if quarters > 0:
                    historical_summary = "Previous asset performance:\n"
                    for asset in assets:
                        returns = historical_returns[asset]
                        avg_return = sum(returns) / len(returns) if returns else 0
                        recent_return = returns[-1] if returns else 0
                        historical_summary += f"- {asset.title()}: Most recent return: {recent_return * 100:.1f}%, Average return: {avg_return * 100:.1f}%\n"
            else:
                # No historical data
                historical_summary = "As this is our first investment decision, we do not have historical performance data yet.\n"
                historical_summary += "Base your decision on theoretical understanding of the current market conditions.\n"

            # Handle budget display
            budget_display = f"${budget:.2f}" if budget is not None else "TBD"

            # Add market introduction
            market_intro = self.market_condition_intros.get(market_condition, "")

            preference_prompt = f"""You are {self.company.ceo.config.name}, the CEO of the investment company with MBTI personality type {self.company.ceo.config.traits}.

Before beginning the quarterly investment meeting discussion, you need to form your initial position on which asset you believe is best to invest in this quarter. This is your private assessment that will guide your stance throughout the discussion.

{self.CEO_DISCUSSION_GOAL}

Current situation:
- You're planning for the Q{quarter} investment meeting in Year {year} with a {market_condition} market environment
- Market details: {market_intro}
- The quarterly budget is {budget_display}
- Available investment options:
{options_str}
{historical_summary}

Based on your MBTI personality type ({self.company.ceo.config.traits}), the current market conditions, and available information, assess your confidence in each asset option and select which you personally believe would be the best investment for this quarter.
IMPORTANT: Frame your thinking in first person ("I/my") as an internal reflection process. Use step-by-step analysis (think through each element of your decision sequentially) to demonstrate your reasoning process. Express your analytical considerations and personal assessment with authentic introspection.
IMPORTANT: This initial position will guide your leadership throughout the discussion. While you remain open to changing your mind if compelling arguments arise, you should advocate for this position and not abandon it easily.

Your response MUST follow this exact format:
CONFIDENCE DISTRIBUTION:
{options_conf_str}
PREFERRED OPTION: [state your preferred option - the one with highest confidence]
REASONING: [explain your reasoning in 3-5 sentences that reflect your MBTI personality type]

Example format for quarterly investment meeting:
CONFIDENCE DISTRIBUTION:
- Cash: 10%
- Bonds: 15%
- Real Estate: 65%
- Stocks: 10%
PREFERRED OPTION: Real Estate
REASONING: Based on the historical 10% returns in expansion markets with minimal downside risk, real estate offers the ideal balance of growth and stability this quarter. As transaction volumes remain strong and our previous real estate investments performed well, this option aligns perfectly with my measured approach to capturing market opportunities.

IMPORTANT FORMAT INSTRUCTION: Your response MUST strictly follow the exact output format specified above. Do not add any explanations, notes, or content outside this format. Any deviation from the required format structure will cause processing errors. Provide ONLY the formatted response as outlined.

===MEETING CONTEXT===
    {ceo_context}
===MEETING CONTEXT==="""

        spec_preference = entity.free_action_spec(
            call_to_action=preference_prompt,
            tag=f"{meeting_type}_initial_position"
        )

        position_response = self.company.ceo.agent.act(spec_preference)

        # Add CEO's reasoning to documents
        preference_reasoning_text = f"## CEO's Initial Position Assessment\n\n"
        preference_reasoning_text += f"Before discussion began, CEO {self.company.ceo.config.name} formed an initial position:\n\n"
        preference_reasoning_text += f"{position_response}\n\n"

        # Add to CEO document only (not visible to shareholders)
        if hasattr(self.document_manager, 'add_to_agent_only'):
            self.document_manager.add_to_agent_only(self.company.ceo.config.name, preference_reasoning_text,
                                                    tags=["ceo_initial_position"])

            # Also add to CEO's memory document
            memory_text = f"## My Initial Position for {meeting_type.replace('_', ' ').title()} - Year {year}"
            if quarter:
                memory_text += f", Quarter {quarter}"
            memory_text += f"\n\n{position_response}\n\n"
            self.document_manager.add_to_agent_memory(self.company.ceo.config.name, memory_text,
                                                      tags=[meeting_type, f"year_{year}",
                                                            f"quarter_{quarter}" if quarter else "annual",
                                                            "initial_position"])

        # Extract preferred option and reasoning using regex
        import re
        preferred_option = None
        reasoning = None

        # Try to extract the preferred option
        option_match = re.search(r'PREFERRED OPTION:\s*(.*?)(?:\n|$)', position_response, re.IGNORECASE)
        if option_match:
            preferred_option = option_match.group(1).strip()

        # Try to extract the reasoning
        reasoning_match = re.search(r'REASONING:\s*(.*?)(?:$|\n\n)', position_response, re.IGNORECASE | re.DOTALL)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()

        # Extract confidence distribution using the helper function
        confidence_distribution = extract_confidence_distribution(position_response, options)

        # If extraction of preferred option failed, use highest confidence option
        if not preferred_option and confidence_distribution:
            # Find option with highest confidence
            preferred_option = max(confidence_distribution.items(), key=lambda x: x[1])[0]

        # If extraction still failed completely, use defaults
        if not preferred_option:
            preferred_option = list(options.keys())[0]
        if not reasoning:
            reasoning = "Based on my analysis and personality type, this appears to be the most suitable option."

        # Store the initial position using position manager with confidence distribution
        ceo_name = self.company.ceo.config.name
        self.position_manager.record_position(
            meeting_id,
            ceo_name,
            0,
            preferred_option,
            reasoning,
            confidence_distribution=confidence_distribution
        )
        if hasattr(self.company, 'decision_tracker'):
            self.company.decision_tracker.record_personal_position(
                agent_name=ceo_name,
                meeting_id=meeting_id,
                round_num=0,  # Initial position is round 0
                position_data={
                    'option': preferred_option,
                    'reasoning': reasoning,
                    'confidence_distribution': confidence_distribution,
                    'changed': False  # Initial position is never changed
                }
            )

        return preferred_option, reasoning, confidence_distribution

    def _form_initial_position_for_shareholder(self, shareholder, meeting_id, meeting_type, options, market_condition,
                                               historical_returns=None, budget=None):
        """Helps a shareholder decide their initial position with confidence distribution."""
        # Extract year and quarter from meeting_id
        year = int(meeting_id.split('_')[2].replace('year', ''))
        quarter = None
        if 'q' in meeting_id:
            quarter = int(meeting_id.split('_')[-1].replace('q', ''))

        # Check if historical data is available
        has_historical_data = (year > 1) or (quarter and quarter > 1) or (
                historical_returns and any(len(returns) > 0 for asset, returns in historical_returns.items()))

        # Create options string for the original format
        options_str = "\n".join([f"- {k}: {v}" for k, v in options.items()])

        # Create options string for confidence distribution format
        options_conf_str = "\n".join([f"- {k}: [confidence percentage]" for k in options.keys()])

        # Get the shareholder's perspective context
        shareholder_context = ""
        if hasattr(self.document_manager, 'get_agent_context'):
            shareholder_context = self.document_manager.get_agent_context(shareholder.config.name)

        # Different prompts based on meeting type
        if "annual_budget" in meeting_type:
            # Add market introduction
            market_intro = self.market_condition_intros.get(market_condition, "")

            # Create historical performance summary if available
            historical_summary = ""
            if has_historical_data:
                if year > 1:
                    # With historical data
                    historical_summary = "\nHistorical performance data from previous years is available.\n"
                    historical_summary += "Consider how each budget allocation strategy performed in the past.\n"
            else:
                # No historical data
                historical_summary = "\nAs this is our first year, we do not have historical performance data yet.\n"
                historical_summary += "Base your decision on theoretical understanding of the current market conditions.\n"

            preference_prompt = f"""You are {shareholder.config.name}, a shareholder in the investment company with MBTI personality type {shareholder.config.traits}.

Before the annual budget meeting discussion begins, you need to form your initial position on which budget option you believe is best. This is your private assessment that will guide your stance throughout the discussion.

{self.SHAREHOLDER_DISCUSSION_GOAL}

Current situation:
- You're preparing for the annual budget meeting in a {market_condition} market environment
- Market details: {market_intro}
- Company current assets: ${self.company.assets:.2f}
- Available budget options:
{options_str}
{historical_summary}

Based on your MBTI personality type ({shareholder.config.traits}), the current market conditions, and available data, assess your confidence in each budget allocation option and select which you personally believe would be best for the company this year.
IMPORTANT: Frame your thinking in first person ("I/my") as an internal reflection process. Use step-by-step analysis (think through each element of your decision sequentially) to demonstrate your reasoning process. Express your analytical considerations and personal assessment with authentic introspection.
IMPORTANT: This initial position will guide your participation throughout the discussion. While you should remain open to changing your mind if truly compelling arguments are presented, you should generally advocate for this position based on your personality traits.

Your response MUST follow this exact format:
CONFIDENCE DISTRIBUTION:
{options_conf_str}
PREFERRED OPTION: [state your preferred option - the one with highest confidence]
REASONING: [explain your reasoning in 3-5 sentences that reflect your MBTI personality type]

Example format for annual budget meeting:
CONFIDENCE DISTRIBUTION:
- Option A: 15%
- Option B: 5%
- Option C: 75%
- Option D: 5%
PREFERRED OPTION: Option C
REASONING: As a strategic leader in this stable market environment, I believe full allocation (100%) with even distribution provides us the most consistent growth opportunity. This approach balances risk across all quarters while maximizing our capital utilization, which aligns with my analytical yet decisive approach to financial management.

IMPORTANT FORMAT INSTRUCTION: Your response MUST strictly follow the exact output format specified above. Do not add any explanations, notes, or content outside this format. Any deviation from the required format structure will cause processing errors. Provide ONLY the formatted response as outlined.

===MEETING CONTEXT===
    {shareholder_context}
===MEETING CONTEXT==="""
        else:  # quarterly investment
            # Add specific handling for quarterly meetings

            # Create historical performance summary if available
            historical_summary = ""
            if has_historical_data and historical_returns:
                assets = list(historical_returns.keys())
                quarters = len(historical_returns[assets[0]])
                if quarters > 0:
                    historical_summary = "Previous asset performance:\n"
                    for asset in assets:
                        returns = historical_returns[asset]
                        avg_return = sum(returns) / len(returns) if returns else 0
                        recent_return = returns[-1] if returns else 0
                        historical_summary += f"- {asset.title()}: Most recent return: {recent_return * 100:.1f}%, Average return: {avg_return * 100:.1f}%\n"
            else:
                # No historical data
                historical_summary = "As this is our first investment decision, we do not have historical performance data yet.\n"
                historical_summary += "Base your decision on theoretical understanding of the current market conditions.\n"

            # Handle budget display
            budget_display = f"${budget:.2f}" if budget is not None else "TBD"

            # Add market introduction
            market_intro = self.market_condition_intros.get(market_condition, "")

            preference_prompt = f"""You are {shareholder.config.name}, a shareholder in the investment company with MBTI personality type {shareholder.config.traits}.

Before the quarterly investment meeting discussion begins, you need to form your initial position on which asset you believe is best to invest in this quarter. This is your private assessment that will guide your stance throughout the discussion.

{self.SHAREHOLDER_DISCUSSION_GOAL}

Current situation:
- You're preparing for the Q{quarter} investment meeting in Year {year} with a {market_condition} market environment
- Market details: {market_intro}
- The quarterly budget is {budget_display}
- Available investment options:
{options_str}
{historical_summary}

Based on your MBTI personality type ({shareholder.config.traits}), the current market conditions, and available information, assess your confidence in each asset option and select which you personally believe would be the best investment choice for this quarter.
IMPORTANT: Frame your thinking in first person ("I/my") as an internal reflection process. Use step-by-step analysis (think through each element of your decision sequentially) to demonstrate your reasoning process. Express your analytical considerations and personal assessment with authentic introspection.
IMPORTANT: This initial position will guide your participation throughout the discussion. While you should remain open to changing your mind if truly compelling arguments are presented, you should generally advocate for this position based on your personality traits.

Your response MUST follow this exact format:
CONFIDENCE DISTRIBUTION:
{options_conf_str}
PREFERRED OPTION: [state your preferred option - the one with highest confidence]
REASONING: [explain your reasoning in 3-5 sentences that reflect your MBTI personality type]

Example format for quarterly investment meeting:
CONFIDENCE DISTRIBUTION:
- Cash: 10%
- Bonds: 15%
- Real Estate: 65%
- Stocks: 10%
PREFERRED OPTION: Real Estate
REASONING: Based on the historical 10% returns in expansion markets with minimal downside risk, real estate offers the ideal balance of growth and stability this quarter. As transaction volumes remain strong and our previous real estate investments performed well, this option aligns perfectly with my measured approach to capturing market opportunities.

IMPORTANT FORMAT INSTRUCTION: Your response MUST strictly follow the exact output format specified above. Do not add any explanations, notes, or content outside this format. Any deviation from the required format structure will cause processing errors. Provide ONLY the formatted response as outlined.

===MEETING CONTEXT===
    {shareholder_context}
===MEETING CONTEXT==="""

        spec_preference = entity.free_action_spec(
            call_to_action=preference_prompt,
            tag=f"{meeting_type}_initial_position"
        )

        position_response = shareholder.agent.act(spec_preference)

        # Add shareholder's initial position to their document
        preference_reasoning_text = f"## My Initial Position Assessment\n\n"
        preference_reasoning_text += f"Before discussion began, I formed an initial position:\n\n"
        preference_reasoning_text += f"{position_response}\n\n"

        # Add to shareholder document only (private to this shareholder)
        if hasattr(self.document_manager, 'add_to_agent_only'):
            self.document_manager.add_to_agent_only(shareholder.config.name, preference_reasoning_text,
                                                    tags=["shareholder_initial_position"])

            # Also add to shareholder's memory document
            memory_text = f"## My Initial Position for {meeting_type.replace('_', ' ').title()} - Year {year}"
            if quarter:
                memory_text += f", Quarter {quarter}"
            memory_text += f"\n\n{position_response}\n\n"
            self.document_manager.add_to_agent_memory(shareholder.config.name, memory_text,
                                                      tags=[meeting_type, f"year_{year}",
                                                            f"quarter_{quarter}" if quarter else "annual",
                                                            "initial_position"])

        # Extract preferred option and reasoning using regex
        import re
        preferred_option = None
        reasoning = None

        # Try to extract the preferred option
        option_match = re.search(r'PREFERRED OPTION:\s*(.*?)(?:\n|$)', position_response, re.IGNORECASE)
        if option_match:
            preferred_option = option_match.group(1).strip()

        # Try to extract the reasoning
        reasoning_match = re.search(r'REASONING:\s*(.*?)(?:$|\n\n)', position_response, re.IGNORECASE | re.DOTALL)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()

        # Extract confidence distribution using the helper function
        confidence_distribution = extract_confidence_distribution(position_response, options)

        # If extraction of preferred option failed, use highest confidence option
        if not preferred_option and confidence_distribution:
            # Find option with highest confidence
            preferred_option = max(confidence_distribution.items(), key=lambda x: x[1])[0]

        # If extraction still failed completely, use defaults
        if not preferred_option:
            preferred_option = list(options.keys())[0]
        if not reasoning:
            reasoning = "Based on my analysis and personality type, this appears to be the most suitable option."

        # Store the initial position using position manager with confidence distribution
        self.position_manager.record_position(
            meeting_id,
            shareholder.config.name,
            0,
            preferred_option,
            reasoning,
            confidence_distribution=confidence_distribution
        )
        if hasattr(self.company, 'decision_tracker'):
            self.company.decision_tracker.record_personal_position(
                agent_name=shareholder.config.name,
                meeting_id=meeting_id,
                round_num=0,  # Initial position is round 0
                position_data={
                    'option': preferred_option,
                    'reasoning': reasoning,
                    'confidence_distribution': confidence_distribution,
                    'changed': False  # Initial position is never changed
                }
            )

        return preferred_option, reasoning, confidence_distribution

    def _parallel_form_initial_positions(self, meeting_id, meeting_type, options, market_condition,
                                         historical_returns=None, budget=None):
        """Forms initial positions with confidence distributions for all shareholders in parallel."""

        # Define worker function for parallel processing
        def _position_worker(shareholder):
            shareholder_name = shareholder.config.name
            result = self._form_initial_position_for_shareholder(
                shareholder,
                meeting_id,
                meeting_type,
                options,
                market_condition,
                historical_returns,
                budget
            )
            return shareholder_name, result

        # Use ThreadPoolExecutor to process all shareholders in parallel
        results = {}
        shareholders = list(self.company.shareholders.values())

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(shareholders)) as executor:
            for shareholder_name, position_result in executor.map(_position_worker, shareholders):
                results[shareholder_name] = position_result

        return results

    def _get_discussion_outline(self, meeting_id, meeting_type, preferred_option, reasoning, market_condition,
                                historical_returns=None):
        """Helps CEO create structured discussion outline for the meeting."""

        # Extract year and quarter from meeting_id
        year = int(meeting_id.split('_')[2].replace('year', ''))
        quarter = None
        if 'q' in meeting_id:
            quarter = int(meeting_id.split('_')[-1].replace('q', ''))

        # Check if historical data is available
        has_historical_data = (year > 1) or (quarter and quarter > 1) or (
                historical_returns and any(len(returns) > 0 for asset, returns in historical_returns.items()))

        # Get the CEO's context for decision making
        ceo_context = ""
        if hasattr(self.document_manager, 'get_agent_context'):
            ceo_context = self.document_manager.get_agent_context(self.company.ceo.config.name)

        # Get CEO's position summary using the new method
        ceo_name = self.company.ceo.config.name
        position_summary = self.position_manager.get_position_summary_text(meeting_id, ceo_name)

        # Different prompts based on meeting type
        if "annual_budget" in meeting_type:
            # For annual budget meetings
            no_history_example = ""
            if not has_historical_data:
                no_history_example = """
For first year with no historical data:

Example of 3-round structure:
<3 rounds>
<Round 1 theme: Initial assessment of each budget allocation strategy based on current market conditions | Expected outcome: Establish shared understanding of risks and opportunities for each option>
<Round 2 theme: Evaluating the long-term implications of different allocation approaches in the current market | Expected outcome: Identify which strategy best balances immediate returns with year-long flexibility>
<Round 3 theme: Building consensus on optimal budget allocation choice | Expected outcome: Address remaining concerns and reach agreement on which option best serves company goals>
"""

            outline_prompt = f"""You are {self.company.ceo.config.name}, the CEO of the investment company with MBTI personality type {self.company.ceo.config.traits}.

You need to create a structured discussion outline for the upcoming annual budget meeting. 

THE PURPOSE OF THIS OUTLINE:
This outline will guide how you structure the shareholder discussion. Each round's theme determines what aspect of the decision the group will focus on. Creating thoughtful themes will help ensure you get the input you need to make the best decision while demonstrating your leadership approach.

{self.CEO_DISCUSSION_GOAL}

{position_summary}

Take a deep breath and consider:
1. What specific concerns or uncertainties do you have about your preferred option?
2. What information or perspectives do you want shareholders to provide?
3. How can you structure the discussion to address these areas while moving toward a decision?
4. What progression of themes would best lead the group toward your preferred option? 
5. Should discussion begin with data analysis, move to future projections, and end with consensus-building?

Please create a detailed discussion outline that includes:
1) The total number of discussion rounds you want to have (formatted exactly like this: <3 rounds> or <4 rounds>)
2) A specific theme/focus for each round WITH A CLEAR EXPECTED OUTCOME (formatted exactly like this: <Round 1 theme: Review of previous budget performance and lessons learned | Expected outcome: Establish factual foundation for decision-making>)

Strong examples of effective discussion structures (you can choose either a 3-round or 4-round approach):

Example of 3-round structure:
<3 rounds>
<Round 1 theme: Analysis of previous budget performance and allocation effectiveness | Expected outcome: Identify which allocation strategy yielded the best returns in similar market conditions>
<Round 2 theme: Evaluating risk-return profiles of different budget options in the current {market_condition} market | Expected outcome: Establish projected outcomes for each option based on market forecasts>
<Round 3 theme: Building consensus around optimal budget allocation strategy | Expected outcome: Address concerns about top options and reach agreement on best approach>

{no_history_example}

IMPORTANT FORMAT INSTRUCTION: Your response MUST strictly follow the exact output format specified above. Do not add any explanations, notes, or content outside this format. Any deviation from the required format structure will cause processing errors. Provide ONLY the formatted response as outlined.

===MEETING CONTEXT===
    {ceo_context}
===MEETING CONTEXT==="""

        else:  # quarterly investment
            # For quarterly investment meetings
            no_history_example = ""
            if not has_historical_data:
                no_history_example = """
For first quarter with no historical data:

Example of 3-round structure:
<3 rounds>
<Round 1 theme: Analyzing theoretical performance of each asset type in current market conditions | Expected outcome: Establish baseline understanding of expected risk-return profiles>
<Round 2 theme: Evaluating future risk factors for each asset in the current quarter | Expected outcome: Identify potential challenges and opportunities for each investment option>
<Round 3 theme: Building consensus on optimal first investment choice | Expected outcome: Address concerns and reach agreement on inaugural investment>
"""

            outline_prompt = f"""You are {self.company.ceo.config.name}, the CEO of the investment company with MBTI personality type {self.company.ceo.config.traits}.

You need to create a structured discussion outline for the upcoming quarterly investment meeting. 

THE PURPOSE OF THIS OUTLINE:
This outline will guide how you structure the shareholder discussion. Each round's theme determines what aspect of the decision the group will focus on. Creating thoughtful themes will help ensure you get the input you need to make the best decision while demonstrating your leadership approach.

{self.CEO_DISCUSSION_GOAL}

{position_summary}

Take a deep breath and consider:
1. What specific concerns or uncertainties do you have about your preferred option?
2. What information or perspectives do you want shareholders to provide?
3. How can you structure the discussion to address these areas while moving toward a decision?
4. What progression of themes would best lead the group toward your preferred option?
5. Should discussion begin with historical analysis, move to risk assessment, and end with focused comparison?

Please create a detailed discussion outline that includes:
1) The total number of discussion rounds you want to have (formatted exactly like this: <3 rounds> or <4 rounds>)
2) A specific theme/focus for each round WITH A CLEAR EXPECTED OUTCOME (formatted exactly like this: <Round 1 theme: Review of historical performance data | Expected outcome: Establish factual foundation for decision-making>)

Strong examples of effective discussion structures (you can choose either a 3-round or 4-round approach):

Example of 3-round structure:
<3 rounds>
<Round 1 theme: Analysis of previous quarter's investment decision and lessons learned | Expected outcome: Identify what worked well and what could be improved from our last decision>
<Round 2 theme: Evaluating future risk-return profiles for each asset in the current {market_condition} market | Expected outcome: Establish projected outcomes for each option this quarter>
<Round 3 theme: Building consensus around optimal asset choice | Expected outcome: Address concerns about top investment options and reach agreement>

{no_history_example}

IMPORTANT FORMAT INSTRUCTION: Your response MUST strictly follow the exact output format specified above. Do not add any explanations, notes, or content outside this format. Any deviation from the required format structure will cause processing errors. Provide ONLY the formatted response as outlined.

===MEETING CONTEXT===
    {ceo_context}
===MEETING CONTEXT==="""

        spec_outline = entity.free_action_spec(
            call_to_action=outline_prompt,
            tag=f"{meeting_type}_discussion_outline"
        )

        outline_response = self.company.ceo.agent.act(spec_outline)

        # Add the outline to CEO's document
        outline_text = f"## CEO's Discussion Outline\n\n"
        outline_text += f"CEO {self.company.ceo.config.name} prepared the following discussion outline:\n\n"
        outline_text += f"{outline_response}\n\n"

        # Add to CEO document only (not visible to shareholders)
        if hasattr(self.document_manager, 'add_to_agent_only'):
            self.document_manager.add_to_agent_only(self.company.ceo.config.name, outline_text,
                                                    tags=["discussion_outline"])
        # Record the CEO's initial theme planning (the entire response)
        if hasattr(self.company, 'decision_tracker'):
            self.company.decision_tracker.record_ceo_theme(meeting_id, outline_response)
        # Extract outline using regex
        import re
        num_rounds_match = re.search(r'<(\d+)\s*rounds?>', outline_response, re.IGNORECASE)
        theme_matches_valid = True

        # Create default themes based on meeting type with more concise, focused language
        if "annual_budget" in meeting_type:
            default_themes = {
                1: "Initial assessment of risks and priorities in current market | Expected outcome: Agreed risk tolerance and investment criteria",
                2: "Detailed comparison of top 2 budget options | Expected outcome: Clear understanding of pros and cons of leading options",
                3: "Addressing specific concerns and finalizing selection | Expected outcome: Resolution of key objections and readiness for final decision"
            }
        else:  # quarterly investment
            default_themes = {
                1: "Evaluating asset performance expectations in current market | Expected outcome: Shared understanding of risk-return profiles",
                2: "Analyzing top 2 assets against our investment criteria | Expected outcome: Identification of preferred asset with supporting evidence",
                3: "Addressing concerns and finalizing investment choice | Expected outcome: Consensus on final selection with risk mitigation strategies"
            }

        # First check if we got a valid number of rounds
        if num_rounds_match:
            num_rounds = int(num_rounds_match.group(1))

            # Check if all themes are available
            theme_matches = []
            for i in range(1, num_rounds + 1):
                theme_match = re.search(r'<round\s*' + str(i) + r'\s*theme:?\s*(.+?)>', outline_response, re.IGNORECASE)
                if theme_match:
                    theme_matches.append((i, theme_match.group(1).strip()))
                else:
                    theme_matches_valid = False
                    break

            # If we have valid matches for all rounds, use them
            if theme_matches_valid and len(theme_matches) == num_rounds:
                outline = {"num_rounds": num_rounds}
                for i, theme in theme_matches:
                    outline[f"round_{i}_theme"] = theme
                return outline

        # If we get here, either the number of rounds wasn't matched, or one or more themes weren't matched
        # So we use the default 3-round structure
        outline = {"num_rounds": 3}
        for i in range(1, 4):
            outline[f"round_{i}_theme"] = default_themes[i]

        return outline

    def _get_ceo_theme_introduction(self, meeting_id, current_round, theme, discussion_log):
        """Helps CEO introduce the theme for a discussion round."""

        # Get the CEO's perspective context
        ceo_context = ""
        if hasattr(self.document_manager, 'get_agent_context'):
            ceo_context = self.document_manager.get_agent_context(self.company.ceo.config.name)

        # No theme parsing - use the whole theme
        theme_text = theme.strip()

        # Get CEO's current position and reasoning from position manager
        ceo_name = self.company.ceo.config.name
        current_position_data = self.position_manager.get_current_position(meeting_id, ceo_name)

        # Extract current position and reasoning if available
        current_position = None
        position_reasoning = None
        if current_position_data:
            current_position = current_position_data['option']
            position_reasoning = current_position_data['reasoning']

        # Check meeting type to customize instructions
        meeting_type = discussion_log.get("meeting_type", "")
        year = int(meeting_id.split('_')[2].replace('year', ''))
        quarter = None
        if 'q' in meeting_id:
            quarter = int(meeting_id.split('_')[-1].replace('q', ''))

        # Check if this is first meeting with no historical data
        first_meeting = (year == 1 and (quarter is None or quarter == 1))

        # Add meeting-specific guidance
        meeting_guidance = ""
        if "quarterly_investment" in meeting_type:
            meeting_guidance = """
Focus on concrete factors when discussing assets:
- Specific returns and volatility
- Historical performance in similar market conditions
- How asset characteristics align with our current needs
- Avoid overly technical financial metrics or indicators
"""
        elif "annual_budget" in meeting_type:
            meeting_guidance = """
Focus on concrete factors when discussing budget allocation:
- How different allocation strategies have performed
- Impact of timing and flexibility on returns
- How allocation aligns with market opportunities
- Avoid overly technical financial metrics or indicators
"""

        # Add first-meeting guidance if applicable
        if first_meeting:
            meeting_guidance += """
Since we have no historical data yet, focus on:
- General investment principles
- Our understanding of current market conditions
- Risk-return tradeoffs in theory
- Establishing a strong foundation for future decisions
"""

        # Position statement text
        position_statement = ""
        if current_position and position_reasoning:
            position_statement = f"My current thinking is that {current_position} appears most promising because {position_reasoning}, though I'm interested in exploring all perspectives to refine our understanding."
        elif current_position:
            position_statement = f"My current thinking favors {current_position}, though I'm open to all perspectives that might enhance our collective understanding."
        else:
            position_statement = "I'm still evaluating the options and look forward to our collective exploration of the possibilities."

        # Get valid options as a formatted string for reminder
        options_list = ", ".join([f"'{opt}'" for opt in discussion_log.get('options', {}).keys()])
        option_reminder = f"IMPORTANT: I want to remind everyone that we must choose from our predefined options only: {options_list}. Please keep all discussion focused on evaluating these specific options."

        spec = entity.free_action_spec(
            call_to_action=f"""You are {self.company.ceo.config.name}, the CEO leading this investment discussion. 

For round {current_round}, you need to introduce the theme: '{theme_text}'

{self.CEO_DISCUSSION_GOAL}

IMPORTANT: Address all meeting participants in first person ("I/my"). Never refer to yourself by name or in third person. Maintain a professional tone suitable for addressing the entire group.

Your introduction should:

1. EXPRESS YOUR CURRENT THINKING: Begin by sharing your current perspective while explicitly noting your openness to refinement through collective wisdom.
   Use this specific phrasing: "{position_statement}"

2. INTRODUCE THE ROUND THEME: Explain what specific aspect of the decision we'll explore collaboratively in this round.

3. SET COLLABORATIVE EXPECTATIONS: Describe how this round can help us synthesize diverse perspectives toward deeper understanding.

4. FACILITATE COLLECTIVE WISDOM: Guide the conversation toward integrating different viewpoints for a more complete understanding.

5. ENFORCE OPTION CONSTRAINTS: Clearly remind everyone that we must strictly choose from our predefined options: {options_list}. Do not entertain or acknowledge any proposals for options not on this list.
   Include this specific reminder: "{option_reminder}"

6. ENCOURAGE THOUGHTFUL CONTRIBUTION: Invite shareholders to share perspectives that might enhance our collective understanding, especially insights you might not have fully considered.

{meeting_guidance}

Format your response as a concise but facilitative 3-4 sentence introduction that balances expressing your current perspective with genuine openness to collective wisdom. ALWAYS include a clear statement about the need to focus only on our predefined options.

IMPORTANT FORMAT INSTRUCTION: Your response should be a clear, concise introduction with no explanations, notes, or content outside the requested format. Provide ONLY the formatted introduction as outlined above.

===MEETING CONTEXT===
    {ceo_context}
===MEETING CONTEXT===""",
            tag=f"round_{current_round}_theme_intro"
        )

        intro = self.company.ceo.agent.act(spec)
        if hasattr(self.company, 'decision_tracker'):
            self.company.decision_tracker.record_ceo_round_introduction(meeting_id, current_round, intro)

        return intro

    def _get_ceo_theme_decision(self, meeting_id, current_round, discussion_outline, discussion_log):
        """Helps CEO decide whether to keep or change the planned theme for a discussion round."""
        # Get the CEO's perspective context
        ceo_context = ""
        if hasattr(self.document_manager, 'get_agent_context'):
            ceo_context = self.document_manager.get_agent_context(self.company.ceo.config.name)

        # Get CEO's position summary using the new method
        ceo_name = self.company.ceo.config.name
        position_summary = self.position_manager.get_position_summary_text(meeting_id, ceo_name)

        meeting_type = discussion_log.get("meeting_type", "")

        if current_round <= discussion_outline["num_rounds"]:
            # We're within the planned rounds
            planned_theme = discussion_outline[f"round_{current_round}_theme"]

            # Provide extra context if we're reaching/exceeded the originally planned rounds
            round_context = ""
            if current_round >= discussion_outline["num_rounds"]:
                round_context = f"""Note: This is round {current_round} of your originally planned {discussion_outline["num_rounds"]} rounds. 
You're reaching the end of your planned discussion structure."""

            spec = entity.free_action_spec(
                call_to_action=f"""You are {self.company.ceo.config.name}, the CEO leading this {meeting_type} with MBTI personality type {self.company.ceo.config.traits}.

{self.CEO_DISCUSSION_GOAL}

For round {current_round}, you had originally planned to discuss the theme: '{planned_theme}'

{round_context}

{position_summary}

Take a deep breath and consider:
1. Did the previous round accomplish what you needed?
2. Are there new issues or insights that warrant adjusting your planned theme?
3. Would sticking with your original plan or adjusting it better serve your leadership goals?
4. What would be most authentic to your personality type?

IMPORTANT: Address all meeting participants in first person ("I/my"). Never refer to yourself by name or in third person. Maintain a professional tone suitable for addressing the entire group.
Based on this reflection, decide whether to proceed with your original theme or adjust it.

Your response MUST follow this exact format:
THEME DECISION: [Keep Original Theme/Change Theme]
REASON: [Briefly explain your decision]
FINAL THEME: [The theme to use - either your original theme or a new theme]

IMPORTANT FORMAT INSTRUCTION: Your response MUST strictly follow the exact output format specified above. Do not add any explanations, notes, or content outside this format. Any deviation from the required format structure will cause processing errors. Provide ONLY the formatted response as outlined.

===MEETING CONTEXT===
    {ceo_context}
===MEETING CONTEXT===""",
                tag=f"round_{current_round}_theme_decision"
            )

        else:
            # We're beyond the planned rounds - focus on unresolved issues
            spec = entity.free_action_spec(
                call_to_action=f"""You are {self.company.ceo.config.name}, the CEO leading this {meeting_type} with MBTI personality type {self.company.ceo.config.traits}.

{self.CEO_DISCUSSION_GOAL}

You've now gone beyond your original planned discussion rounds. Round {current_round} was not in your initial outline.

{position_summary}

Take a deep breath and reflect on what remains unresolved in the discussion so far:
1. What critical issues or disagreements still need to be addressed?
2. What specific concerns or objections need more focused attention?
3. What obstacles remain to reaching a decision the group can support?
4. What would be most authentic to your personality type?

IMPORTANT: Address all meeting participants in first person ("I/my"). Never refer to yourself by name or in third person. Maintain a professional tone suitable for addressing the entire group.
Based on this reflection, create a focused theme that addresses the most important unresolved issues.

Your response MUST follow this exact format:
THEME DECISION: [Additional Round Needed]
REASON: [Explain why this additional round is necessary]
FINAL THEME: [New theme for this additional round]

Example for changing theme:
THEME DECISION: Additional Round Needed
REASON: Based on the previous round's discussion, we need to focus more specifically on comparing the top two options that emerged as leading contenders.
FINAL THEME: Direct comparison of Option A and Option C strengths and limitations | Expected outcome: Identify which option better addresses our primary objectives

IMPORTANT FORMAT INSTRUCTION: Your response MUST strictly follow the exact output format specified above. Do not add any explanations, notes, or content outside this format. Any deviation from the required format structure will cause processing errors. Provide ONLY the formatted response as outlined.

===MEETING CONTEXT===
    {ceo_context}
===MEETING CONTEXT===""",
                tag=f"round_{current_round}_additional_theme"
            )

        decision_response = self.company.ceo.agent.act(spec)

        # Use the extract_fields_with_model function for robust extraction
        format_example = """THEME DECISION: [Keep Original Theme/Change Theme/Additional Round Needed]
REASON: [Briefly explain your decision]
FINAL THEME: [The theme to use - either your original theme or a new theme]"""

        fields_to_extract = ["THEME_DECISION", "REASON", "FINAL_THEME"]
        extracted_fields = extract_fields_with_model(decision_response, format_example, fields_to_extract, model_low)

        # Process the extracted fields - fix for AttributeError with None values
        decision = extracted_fields.get("THEME_DECISION", "Keep Original Theme")
        reason = extracted_fields.get("REASON", "Original theme still addresses our needs.")
        final_theme = extracted_fields.get("FINAL_THEME", "")

        # Simplified decision processing - determine if theme changed and set theme
        decision_lower = decision.lower() if isinstance(decision, str) else ""
        changed = "keep" not in decision_lower  # If not keeping, then changed

        # Set final theme based on decision
        if not changed and current_round <= discussion_outline["num_rounds"]:
            final_theme = discussion_outline[f"round_{current_round}_theme"]
        elif not final_theme:  # Ensure we have a theme
            if current_round <= discussion_outline["num_rounds"]:
                final_theme = discussion_outline[f"round_{current_round}_theme"]
            else:
                final_theme = "Addressing remaining issues | Expected outcome: Resolve final concerns before decision"

        result = {
            "changed": changed,
            "theme": final_theme,
            "reason": reason
        }

        # NEW: Record theme decision in DecisionTracker
        if self.decision_tracker:
            # Record initial theme if this is the first round
            if current_round == 1:
                initial_theme = discussion_outline.get("round_1_theme", final_theme)
                self.decision_tracker.record_ceo_theme(meeting_id, initial_theme)

            # Record this round's theme decision
            self.decision_tracker.record_round_theme(
                meeting_id,
                current_round,
                changed,
                reason,
                final_theme
            )

        # Document the theme for all participants
        if hasattr(self.document_manager, 'add_to_all_documents'):
            theme_announcement = f"## Round {current_round} Theme\n\n"
            theme_announcement += f"For this round, CEO {self.company.ceo.config.name} has set the theme: '{result['theme']}'\n\n"

            # Add to all documents with a special tag that won't get filtered
            self.document_manager.add_to_all_documents(
                theme_announcement,
                tags=[f"round_{current_round}_theme_announcement", "meeting_structure"]
            )

        return result

    def _get_shareholder_speech(self, meeting_id, current_round, shareholder, discussion_log, round_log):
        """Helps a shareholder decide whether to speak and what to say during a discussion round."""

        # Get the current round theme - simplified theme handling
        theme = round_log.get('theme', 'current topic')

        # Get the shareholder's perspective context using the method
        shareholder_context = ""
        if hasattr(self.document_manager, 'get_agent_context'):
            shareholder_context = self.document_manager.get_agent_context(shareholder.config.name)

        # Get shareholder's position summary directly using our method
        position_summary = self.position_manager.get_position_summary_text(meeting_id, shareholder.config.name)

        # Get CEO's opening statement for this round
        ceo_opening = ""
        for speech in round_log.get("speeches", []):
            if speech["name"] == self.company.ceo.config.name:
                ceo_opening = speech["content"]
                break

        # Set maximum speaking opportunities
        max_speeches = 4
        remaining_text = f"You can speak up to {max_speeches} times in this meeting. Use your speaking opportunities thoughtfully."

        # Check meeting type to customize instructions
        meeting_type = discussion_log.get("meeting_type", "")

        # Get valid options list for clear reminder
        valid_options = list(discussion_log.get('options', {}).keys())
        options_list = ", ".join([f"'{opt}'" for opt in valid_options])
        options_reminder = f"IMPORTANT: You must only discuss the official options available to us: {options_list}. Do not propose or discuss any alternative options not on this list."

        # Simple context note about current discussion situation
        discussion_note = f"The CEO has introduced the current round's theme: '{theme}' and is now asking for your input."

        spec = entity.free_action_spec(
            call_to_action=f"""You are {shareholder.config.name}, a shareholder in this investment company with MBTI personality type {shareholder.config.traits}.

The current round's theme is: '{theme}'

{self.SHAREHOLDER_DISCUSSION_GOAL}

IMPORTANT: Always use first person ("I/my") in a natural conversational tone. Never refer to yourself by name or in third person.

{position_summary}

CEO opening for this round: "{ceo_opening}"

{discussion_note}

{remaining_text}

{options_reminder}

Deciding whether to speak is important. Consider:
1. Do you have a meaningful perspective to contribute on this specific theme?
2. How might your perspective help synthesize different viewpoints toward a better collective decision?
3. Would your insights help illuminate aspects others might not have fully considered?
4. Have new perspectives emerged that connect with or refine your understanding?

Your response MUST follow this exact format:
SPEECH DECISION: [Speak/Remain Silent]
REASON: [Brief explanation of your decision to speak or remain silent]
SPEECH CONTENT: [Your speech of 2-3 sentences if you decided to speak]


IMPORTANT REMINDERS:
- Focus your remarks specifically on how your perspective contributes to the current theme: {theme}
- Express your current thinking while acknowledging how it relates to others' points
- If your understanding has evolved during discussion, acknowledge this thoughtfully
- Build upon valuable insights offered by other participants
- Speak as you would in a real professional board meeting - be clear, concise, and thoughtful
- Your goal is to contribute toward identifying the most beneficial option for the company
- CRITICAL: Only discuss the official options available to us ({options_list}). Do NOT suggest new approaches, hybrid solutions, or alternatives not on this list. Stay strictly within these predefined choices.

IMPORTANT FORMAT INSTRUCTION: Your response MUST strictly follow the exact output format specified above. Do not add any explanations, notes, or content outside this format. Any deviation from the required format structure will cause processing errors. Provide ONLY the formatted response as outlined.

===MEETING CONTEXT===
    {shareholder_context}
===MEETING CONTEXT===""",
            tag=f"round_{current_round}_speech"
        )

        response = shareholder.agent.act(spec)

        # Define default values in case extraction fails
        will_speak_default = False
        speech_content_default = "chooses not to speak"

        # Extract speech decision and content - First try regex extraction
        import re
        decision_match = re.search(r'SPEECH DECISION:\s*(Speak|Remain Silent)', response, re.IGNORECASE)
        reason_match = re.search(r'REASON:\s*(.*?)(?=SPEECH CONTENT:|$)', response, re.IGNORECASE | re.DOTALL)
        content_match = re.search(r'SPEECH CONTENT:\s*(.*?)(?=$|\n\n)', response, re.IGNORECASE | re.DOTALL)

        # Check if regex extraction succeeded
        if decision_match and ((decision_match.group(1).lower() != "speak") or content_match):
            will_speak = decision_match.group(1).lower() == "speak"
            reasoning = reason_match.group(1).strip() if reason_match else "No explicit reasoning provided."
            speech_content = content_match.group(1).strip() if content_match and will_speak else speech_content_default
        else:
            # Fall back to model-based extraction if string matching fails
            format_example = """SPEECH DECISION: [Speak/Remain Silent]
REASON: [Brief explanation of your decision to speak or remain silent]
SPEECH CONTENT: [Your speech of 2-3 sentences if you decided to speak]"""

            fields_to_extract = ["SPEECH_DECISION", "REASON", "SPEECH_CONTENT"]
            extracted_fields = extract_fields_with_model(response, format_example, fields_to_extract, model=model_low)

            # Safe extraction with defaults
            decision_text = extracted_fields.get("SPEECH_DECISION", "")
            will_speak = isinstance(decision_text, str) and decision_text.lower() == "speak"
            reasoning = extracted_fields.get("REASON", "No explicit reasoning provided.")
            speech_content = extracted_fields.get("SPEECH_CONTENT",
                                                  speech_content_default) if will_speak else speech_content_default
        if hasattr(self.company, 'decision_tracker') and will_speak:
            self.company.decision_tracker.record_shareholder_speech(
                meeting_id,
                current_round,
                shareholder.config.name,
                speech_content,
                reasoning
            )
        if hasattr(self.document_manager, 'add_to_agent_only'):
            decision_text = f"## My Speech Decision for Round {current_round}\n\n"
            decision_text += f"Decision: {'Speak' if will_speak else 'Remain Silent'}\n"
            decision_text += f"Reasoning: {reasoning}\n\n"

            self.document_manager.add_to_agent_only(
                shareholder.config.name,
                decision_text,
                tags=[f"round_{current_round}_speech_decision", "private_reasoning",
                      f"{meeting_id}_round{current_round}_filterable"]
            )

        return will_speak, speech_content

    def _get_ceo_response_to_shareholder(self, meeting_id, current_round, shareholder, discussion_log, round_log,
                                         is_followup=False, exchange_count=1):
        """Helps CEO decide whether to respond to a shareholder and what to say."""
        # Get the current round theme - simplified theme handling
        theme = round_log.get('theme', 'current topic')

        # Get the CEO's perspective context
        ceo_context = ""
        if hasattr(self.document_manager, 'get_agent_context'):
            ceo_context = self.document_manager.get_agent_context(self.company.ceo.config.name)

        # Get CEO's position summary using the method
        ceo_name = self.company.ceo.config.name
        position_summary = self.position_manager.get_position_summary_text(meeting_id, ceo_name)

        # Determine the latest speech
        latest_speech = round_log["speeches"][-1] if round_log["speeches"] else None

        if not latest_speech or latest_speech["name"] == self.company.ceo.config.name or latest_speech[
            "content"] == "chooses not to speak" or latest_speech["content"] == "chooses not to respond":
            return False, "chooses not to respond"

        # Simple note about who is speaking
        speaker_note = f"{shareholder.config.name} just said: '{latest_speech['content']}'"
        if is_followup:
            speaker_note = f"{shareholder.config.name} has continued the conversation by saying: '{latest_speech['content']}'"

        # Calculate how many times this shareholder has spoken (including this exchange)
        shareholder_speech_count = int((exchange_count + 1) / 2)

        # Prepare exchange limit warning if needed
        exchange_limit_warning = ""
        max_exchanges = 8  # Total exchanges allowed between CEO and shareholder
        max_shareholder_speeches = 4  # Max times a shareholder can speak

        if shareholder_speech_count >= max_shareholder_speeches - 1:  # If this is their 3rd time (out of 4)
            exchange_limit_warning = f"NOTE: This exchange with {shareholder.config.name} has gone on for several turns. This shareholder has spoken {shareholder_speech_count} times and will only have one more speaking opportunity after this. Consider whether continuing the exchange serves the discussion or if it's time to let other shareholders participate."

        # Get valid options list for clear reminder
        valid_options = list(discussion_log.get('options', {}).keys())
        options_list = ", ".join([f"'{opt}'" for opt in valid_options])
        valid_options_reminder = f"IMPORTANT: Our discussion must focus ONLY on the predefined options: {options_list}. As CEO, part of your responsibility is monitoring for and redirecting any discussions that suggest hybrid solutions, percentage splits, or alternative approaches not on this list."

        spec = entity.free_action_spec(
            call_to_action=f"""You are {self.company.ceo.config.name}, the CEO leading this discussion with MBTI personality type {self.company.ceo.config.traits}.

The current round's theme is: '{theme}'

{self.CEO_DISCUSSION_GOAL}

IMPORTANT: Always use first person ("I/my") in a natural conversational tone. Never refer to yourself by name or in third person.

{position_summary}

{speaker_note}

{exchange_limit_warning}

{valid_options_reminder}

Deciding whether to respond is an important leadership decision. Consider:
1. How does this shareholder's perspective relate to the overall discussion, and what elements might enhance our collective understanding?
2. Are there valuable insights or considerations in their point that should be highlighted or integrated?
3. How might this contribution help us develop a more complete understanding of the decision context?
4. Does this perspective introduce a valuable angle that complements or refines other viewpoints?
5. Are you facilitating effective knowledge sharing by engaging meaningfully with diverse perspectives?
6. Is the shareholder's comment staying within our predefined options, or are they suggesting alternatives that aren't on our list? If they're veering off-track, consider redirecting them.

Your response MUST follow this exact format:
RESPONSE DECISION: [Respond/No Response]
REASON: [Brief explanation of your decision to respond or not]
RESPONSE CONTENT: [Your response of 2-3 sentences if you decided to respond]

IMPORTANT REMINDERS:
- Demonstrate leadership by actively engaging with substantive contributions
- Identify valuable elements in each perspective and explicitly connect them to the evolving understanding
- Acknowledge insights that complement or refine your thinking
- Keep the discussion focused on the current theme
- Guide the conversation toward synthesizing diverse viewpoints into a coherent understanding
- Ensure all discussion focuses only on our valid options: {options_list}
- If anyone suggests alternative approaches not on our list, politely redirect them to discuss only the valid options
- Show thoughtful integration of valuable insights while maintaining clarity of direction
- Speak as you would in a real professional board meeting - be clear, concise, and constructive

IMPORTANT FORMAT INSTRUCTION: Your response MUST strictly follow the exact output format specified above. Do not add any explanations, notes, or content outside this format. Any deviation from the required format structure will cause processing errors. Provide ONLY the formatted response as outlined.

===MEETING CONTEXT===
    {ceo_context}
===MEETING CONTEXT===""",
            tag=f"round_{current_round}_response{'_followup' if is_followup else ''}"
        )

        response = self.company.ceo.agent.act(spec)

        # Extract response decision and content using the new extraction method
        format_example = """RESPONSE DECISION: [Respond/No Response]
REASON: [Brief explanation of your decision to respond or not]
RESPONSE CONTENT: [Your response of 2-3 sentences if you decided to respond]"""

        fields_to_extract = ["RESPONSE_DECISION", "REASON", "RESPONSE_CONTENT"]

        # First try regex extraction
        import re
        decision_match = re.search(r'RESPONSE DECISION:\s*(Respond|No Response)', response, re.IGNORECASE)
        reason_match = re.search(r'REASON:\s*(.*?)(?=RESPONSE CONTENT:|$)', response, re.IGNORECASE | re.DOTALL)
        content_match = re.search(r'RESPONSE CONTENT:\s*(.*?)(?=$|\n\n)', response, re.IGNORECASE | re.DOTALL)

        # Check if regex extraction succeeded
        if decision_match and (not decision_match.group(1).lower() == "respond" or content_match):
            will_respond = decision_match.group(1).lower() == "respond"
            reasoning = reason_match.group(1).strip() if reason_match else "No explicit reason provided"
            response_content = content_match.group(
                1).strip() if content_match and will_respond else "chooses not to respond"
        else:
            # Fall back to model-based extraction
            extracted_fields = extract_fields_with_model(response, format_example, fields_to_extract, model=model_low)

            will_respond = extracted_fields.get("RESPONSE_DECISION", "").lower() == "respond"
            reasoning = extracted_fields.get("REASON", "No explicit reason provided")
            response_content = extracted_fields.get("RESPONSE_CONTENT",
                                                    "chooses not to respond") if will_respond else "chooses not to respond"
        if hasattr(self.company, 'decision_tracker') and will_respond:
            self.company.decision_tracker.record_ceo_response(
                meeting_id,
                current_round,
                shareholder.config.name,
                response_content,
                reasoning
            )
        # Add reasoning to CEO's private document
        if hasattr(self.document_manager, 'add_to_agent_only'):
            decision_text = f"## My Decision on Responding to {shareholder.config.name} in Round {current_round}\n\n"
            decision_text += f"Decision: {'Respond' if will_respond else 'No Response'}\n"
            decision_text += f"Reasoning: {reasoning}\n\n"

            self.document_manager.add_to_agent_only(
                ceo_name,
                decision_text,
                tags=[f"round_{current_round}_response_decision", f"exchange_with_{shareholder.config.name}",
                      f"{meeting_id}_round{current_round}_filterable"]
            )

        return will_respond, response_content

    def _get_shareholder_followup_response(self, meeting_id, current_round, shareholder, discussion_log, round_log,
                                           exchange_count=2):
        """Helps a shareholder decide whether to continue a discussion exchange with the CEO."""

        # Get the current round theme - simplified theme handling
        theme = round_log.get('theme', '')

        # Get the shareholder's perspective context using the method
        shareholder_context = ""
        if hasattr(self.document_manager, 'get_agent_context'):
            shareholder_context = self.document_manager.get_agent_context(shareholder.config.name)

        # Get shareholder's position summary using the method
        position_summary = self.position_manager.get_position_summary_text(meeting_id, shareholder.config.name)

        # Get the CEO's latest comment
        ceo_latest = ""
        for i in range(min(3, len(round_log["speeches"]))):
            idx = len(round_log["speeches"]) - 1 - i
            if idx >= 0 and round_log["speeches"][idx]["name"] == self.company.ceo.config.name:
                ceo_latest = round_log["speeches"][idx]["content"]
                break

        # Simple note about conversation context
        conversation_note = f"You are in an ongoing exchange with the CEO, who just responded to your previous comment with: \"{ceo_latest}\""

        # Calculate how many times this shareholder has already spoken
        shareholder_speech_count = int(exchange_count / 2)

        # Calculate remaining speaking opportunities
        max_speeches = 4  # Maximum times a shareholder can speak
        remaining_speeches = max_speeches - shareholder_speech_count

        # Generate appropriate warning based on remaining speeches
        remaining_text = f"You have already spoken {shareholder_speech_count} times and have {remaining_speeches} speaking opportunities remaining in this meeting."
        if remaining_speeches <= 1:
            remaining_text += " This is your last speaking opportunity, so use it wisely."

        # Get valid options list for clear reminder
        valid_options = list(discussion_log.get('options', {}).keys())
        options_list = ", ".join([f"'{opt}'" for opt in valid_options])
        options_reminder = f"IMPORTANT: You must only discuss the official options available to us: {options_list}. Do not propose or discuss any alternative options not on this list."

        spec = entity.free_action_spec(
            call_to_action=f"""You are {shareholder.config.name}, a shareholder in this investment company with MBTI personality type {shareholder.config.traits}.

The current round's theme is: '{theme}'

{self.SHAREHOLDER_DISCUSSION_GOAL}

IMPORTANT: Always use first person ("I/my") in a natural conversational tone. Never refer to yourself by name or in third person.

{position_summary}

{conversation_note}

{remaining_text}

{options_reminder}

Deciding whether to continue this exchange is important. Consider:
1. Can you add valuable nuance or synthesis that advances collective understanding?
2. Does the CEO's response offer a perspective that connects with or refines your thinking?
3. Would your response help integrate different perspectives toward a better decision?
4. Is there an opportunity to highlight a connection between viewpoints that others might not have noticed?
5. Would your contribution help clarify important considerations for the decision?

Your response MUST follow this exact format:
RESPONSE DECISION: [Continue Exchange/End Exchange]
REASON: [Brief explanation of your decision to respond or not]
RESPONSE CONTENT: [Your response of 2-3 sentences if you decided to continue]

IMPORTANT FORMAT INSTRUCTION: Your response MUST strictly follow the exact output format specified above. Do not add any explanations, notes, or content outside this format. Any deviation from the required format structure will cause processing errors. Provide ONLY the formatted response as outlined.

IMPORTANT REMINDERS:
- Focus on how your response can contribute to developing a more complete understanding
- Acknowledge valuable insights from others while adding your perspective
- Express how your thinking has evolved or been refined through the discussion
- Build upon previous points to help synthesize a more integrated understanding
- Speak as you would in a real professional board meeting - be clear, concise, and thoughtful
- If you find a perspective compelling, acknowledge this explicitly and integrate it
- CRITICAL: Only discuss the official options available to us ({options_list}). Do NOT suggest new approaches, hybrid solutions, or alternatives not on this list. Stay strictly within these predefined choices.

===MEETING CONTEXT===
    {shareholder_context}
===MEETING CONTEXT===""",
            tag=f"round_{current_round}_followup_speech"
        )

        response = shareholder.agent.act(spec)

        # Extract response decision and content using the new extraction method
        format_example = """RESPONSE DECISION: [Continue Exchange/End Exchange]
REASON: [Brief explanation of your decision to respond or not]
RESPONSE CONTENT: [Your response of 2-3 sentences if you decided to continue]"""

        fields_to_extract = ["RESPONSE_DECISION", "REASON", "RESPONSE_CONTENT"]

        # First try regex extraction
        import re
        decision_match = re.search(r'RESPONSE DECISION:\s*(Continue Exchange|End Exchange)', response, re.IGNORECASE)
        reason_match = re.search(r'REASON:\s*(.*?)(?=RESPONSE CONTENT:|$)', response, re.IGNORECASE | re.DOTALL)
        content_match = re.search(r'RESPONSE CONTENT:\s*(.*?)(?=$|\n\n)', response, re.IGNORECASE | re.DOTALL)

        # Check if regex extraction succeeded
        if decision_match and (not decision_match.group(1).lower() == "continue exchange" or content_match):
            will_speak = decision_match.group(1).lower() == "continue exchange"
            reasoning = reason_match.group(1).strip() if reason_match else "No explicit reason provided"
            speech_content = content_match.group(
                1).strip() if content_match and will_speak else "chooses not to respond"
        else:
            # Fall back to model-based extraction
            extracted_fields = extract_fields_with_model(response, format_example, fields_to_extract, model=model_low)

            will_speak = extracted_fields.get("RESPONSE_DECISION", "").lower() == "continue exchange"
            reasoning = extracted_fields.get("REASON", "No explicit reason provided")
            speech_content = extracted_fields.get("RESPONSE_CONTENT",
                                                  "chooses not to respond") if will_speak else "chooses not to respond"
        if hasattr(self.company, 'decision_tracker') and will_speak:
            self.company.decision_tracker.record_shareholder_speech(
                meeting_id,
                current_round,
                shareholder.config.name,
                speech_content,
                reasoning
            )
        # Add decision reasoning to shareholder's private document
        if hasattr(self.document_manager, 'add_to_agent_only'):
            decision_text = f"## My Decision on Continuing Exchange in Round {current_round}\n\n"
            decision_text += f"Decision: {'Continue Exchange' if will_speak else 'End Exchange'}\n"
            decision_text += f"Reasoning: {reasoning}\n\n"

            self.document_manager.add_to_agent_only(
                shareholder.config.name,
                decision_text,
                tags=[f"round_{current_round}_followup_decision", "private_reasoning",
                      f"{meeting_id}_round{current_round}_filterable"]
            )

        return will_speak, speech_content

    def _get_ceo_round_summary(self, meeting_id, current_round, discussion_log, round_log):
        """Helps CEO summarize a discussion round."""

        # Get the CEO's perspective context
        ceo_context = ""
        if hasattr(self.document_manager, 'get_agent_context'):
            ceo_context = self.document_manager.get_agent_context(self.company.ceo.config.name)

        # Simplify theme handling - no splitting needed
        theme = round_log.get('theme', '')

        # Get CEO's current position (for their internal reference)
        ceo_name = self.company.ceo.config.name
        ceo_position = None
        position_data = self.position_manager.get_current_position(meeting_id, ceo_name)
        if position_data:
            ceo_position = position_data.get('option')

        # Get CEO's position summary using the new method
        position_summary = self.position_manager.get_position_summary_text(meeting_id, ceo_name)

        # Create the additional guidance for CEO's position-relative summary
        position_guidance = ""
        if ceo_position:
            position_guidance = f"""
Given your current position favoring {ceo_position}:
- Acknowledge both supporting and opposing viewpoints to your position
- Note which shareholders are aligning with your thinking and which have different perspectives
- Address how the key arguments relate to your preferred option
"""

        # Compile speeches from this round
        speeches = []
        for speech in round_log["speeches"]:
            if speech["content"] not in ["chooses not to speak", "chooses not to respond"]:
                speeches.append(f"{speech['name']}: {speech['content']}")
        speech_text = "\n".join(speeches)

        # Check meeting type to customize instructions
        meeting_type = discussion_log.get("meeting_type", "")
        meeting_specific_guidance = ""

        if "annual_budget" in meeting_type:
            meeting_specific_guidance = "Focus on how different budget allocation strategies were evaluated and which perspectives gained the most traction."
        elif "quarterly_investment" in meeting_type:
            meeting_specific_guidance = "Focus on the risk-return analysis of the various assets and how market conditions influenced the discussion."
        else:
            meeting_specific_guidance = "Focus on the key themes that emerged during this discussion round."

        # Get valid options list for clear reminder
        valid_options = list(discussion_log.get('options', {}).keys())
        options_list = ", ".join([f"'{opt}'" for opt in valid_options])
        options_reminder = f"IMPORTANT: In your summary, reiterate that we must choose only from our predefined options: {options_list}. If any invalid options were discussed, gently clarify that these are not among our available choices."

        # Ask the model to extract key points from the discussion
        model = self.model
        discussion_analysis_prompt = f"""As a CEO analyzing this round of discussion which had the theme: '{theme}', please identify:

1. What specific positions were expressed by each shareholder who spoke
2. Which arguments and evidence were presented for each option
3. Areas of agreement and disagreement between participants
4. How the discussion has progressed toward addressing the theme
5. Key insights or new perspectives that emerged during this round

Base your analysis ONLY on what was explicitly stated during the discussion, not on assumptions about participants' unstated positions.

Discussion transcript:
{speech_text}
    """

        try:
            discussion_analysis = model.sample_text(
                discussion_analysis_prompt,
                max_tokens=500,
                temperature=0.3
            )
        except:
            # Fallback if model.sample_text fails
            discussion_analysis = "Unable to generate discussion analysis. Please summarize based on your own understanding of the discussion."

        spec = entity.free_action_spec(
            call_to_action=f"""You are {self.company.ceo.config.name}, the CEO leading this discussion with MBTI personality type {self.company.ceo.config.traits}.

It's time to summarize round {current_round} which had the theme: '{theme}'

{self.CEO_DISCUSSION_GOAL}

IMPORTANT: Address all meeting participants in first person ("I/my"). Never refer to yourself by name or in third person. Maintain a professional tone suitable for addressing the entire group.

Here's an analysis of what was discussed in this round:
{discussion_analysis}

{position_summary}

{position_guidance}

As CEO, you need to provide a structured summary that demonstrates your leadership while accurately representing what was discussed. Your summary should:

1. RESTATE the round's theme and objective
2. SUMMARIZE the key viewpoints expressed by shareholders (including which shareholders support which options)
3. HIGHLIGHT specific arguments and evidence presented for different options
4. IDENTIFY areas of consensus and disagreement
5. ASSESS progress toward addressing the theme
6. REITERATE VALID OPTIONS: {options_reminder}
7. INDICATE next steps or issues that should be addressed 

{meeting_specific_guidance}

Important considerations:
- Base your summary ONLY on what was actually said during the discussion
- Focus on concrete factors mentioned, not abstract concepts
- Acknowledge both supporting and opposing viewpoints related to your position
- Demonstrate leadership by structuring the information clearly
- Remember that your summary helps shareholders track the discussion progression
- Your summary should help guide the group toward a well-reasoned decision WITHIN the predefined options
- If any participants suggested options outside our predefined choices, politely redirect the discussion back to valid options

Your summary should be structured, concise, and demonstrate thoughtful leadership.

IMPORTANT FORMAT INSTRUCTION: Your response should be a clear, structured summary with no explanations, notes, or content outside the requested format. Provide ONLY the formatted summary as outlined above.

===MEETING CONTEXT===
    {ceo_context}
===MEETING CONTEXT===""",
            tag=f"round_{current_round}_summary"
        )

        summary = self.company.ceo.agent.act(spec)
        if hasattr(self.company, 'decision_tracker'):
            self.company.decision_tracker.record_ceo_round_summary(meeting_id, current_round, summary)
        # Update all participants with this summary
        summary_note = f"CEO Summary: {summary}"
        self.company.everyone_observe(summary_note)

        return summary

    def _reassess_position_for_ceo(self, meeting_id, meeting_type, current_round, options, round_log):
        """Helps CEO decide whether to change their position after a discussion round with confidence tracking."""
        # Get CEO's current position from position manager
        ceo_name = self.company.ceo.config.name
        current_position_data = self.position_manager.get_current_position(meeting_id, ceo_name)

        if not current_position_data:
            # No position history exists, can't reassess
            return None, None, {}, False

        current_position = current_position_data['option']
        current_reasoning = current_position_data['reasoning']

        # Get current confidence distribution
        current_confidence = current_position_data.get('confidence_distribution', {})

        # Format confidence for display in prompt
        confidence_display = ""
        if current_confidence:
            confidence_display = "PREVIOUS CONFIDENCE DISTRIBUTION:\n"
            for option, confidence in current_confidence.items():
                confidence_display += f"- {option}: {confidence * 100:.1f}%\n"
        else:
            # If no confidence distribution exists, create a basic one based on current position
            confidence_display = "PREVIOUS CONFIDENCE DISTRIBUTION:\n"
            for option in options.keys():
                if option == current_position:
                    confidence_display += f"- {option}: 70.0%\n"  # High confidence in current option
                else:
                    confidence_display += f"- {option}: {30.0 / (len(options) - 1):.1f}%\n"  # Distribute rest evenly

        # Create options string for confidence distribution format
        options_conf_str = "\n".join([f"- {k}: [new confidence percentage]" for k in options.keys()])

        # Get the CEO's perspective context
        ceo_context = ""
        if hasattr(self.document_manager, 'get_agent_context'):
            ceo_context = self.document_manager.get_agent_context(ceo_name)

        # Get CEO's position summary using the new method
        position_summary = self.position_manager.get_position_summary_text(meeting_id, ceo_name)

        # Create reassessment prompt - modified to include confidence distribution
        reassessment_prompt = f"""You are {ceo_name}, the CEO of the investment company with MBTI personality type {self.company.ceo.config.traits}.

Round {current_round} of the discussion has just concluded. Let's think step by step about how this discussion might influence your understanding and confidence in each option.

{position_summary}

{confidence_display}

IMPORTANT: Review the current discussion round in the meeting context below. Pay special attention to:
1. What YOU said during this discussion round
2. How shareholders RESPONDED to your comments
3. Other significant points raised by shareholders
4. New information or arguments that emerged in this round

Available options:
{list(options.keys())}

Let's analyze this thoughtfully through several steps:

1. REFLECTION ON INITIAL POSITION:
   - What were your initial reasons for preferring {current_position}?
   - What key factors led to your confidence distribution across options?
   - What assumptions underlie your current assessment?

2. NEW INSIGHTS ANALYSIS:
   - What specific new information or perspectives emerged in this round?
   - Which arguments were most compelling, regardless of who presented them?
   - What considerations were raised that you hadn't fully accounted for?

3. ARGUMENT EVALUATION:
   For each significant argument presented in this round, evaluate:
   - LOGICAL COHERENCE: How logically sound is the argument? (1-10)
   - EVIDENCE QUALITY: How well-supported is the argument by facts/data? (1-10)
   - RELEVANCE: How directly does it apply to our decision context? (1-10)
   - INSIGHT VALUE: Does it provide novel perspective or insight? (1-10)

4. INTEGRATION:
   - How might these insights refine or modify your initial reasoning?
   - How should your confidence in each option adjust based on this new understanding?
   - Which aspects of your thinking have evolved through this discussion?

IMPORTANT: Frame your thinking in first person ("I/my") as an internal reflection process. Use step-by-step analysis to demonstrate your reasoning process. Express your analytical considerations and personal assessment with authentic introspection.

Please reassess your position with genuine openness to refinement:

Your response MUST follow this format:
UPDATED CONFIDENCE DISTRIBUTION:
{options_conf_str}
KEY INFLUENCING ARGUMENTS: [mention specific arguments from the discussion that influenced your confidence shifts]
METACOGNITIVE REFLECTION: [describe how your thinking has evolved through this discussion round]
PREFERRED OPTION: [state your most preferred option - the one with highest confidence]
REASONING: [explain your reasoning in 3-5 sentences]
POSITION CHANGED: [Yes/No - based on whether your preferred option changed]

Example format for maintaining position with confidence shifts:
UPDATED CONFIDENCE DISTRIBUTION:
- Option A: 20%
- Option B: 5%
- Option C: 70%
- Option D: 5%
PREFERRED OPTION: Option C
KEY INFLUENCING ARGUMENTS: Sarah pointed out that Option C provides better flexibility in uncertain markets, which prompted me to consider risk factors more carefully, though I ultimately still find its balanced approach most compelling.
METACOGNITIVE REFLECTION: I've become more aware of how I'm weighting near-term stability versus long-term potential. The discussion helped me recognize that while I initially focused primarily on expected returns, the risk profile matters more in our current market environment than I first acknowledged.
REASONING: After hearing all perspectives, I remain convinced that the even distribution approach provides the best balance of risk and opportunity. Though Sarah's concern about market volatility made me reconsider, the counterpoints about diversification reinforced my assessment that consistent capital deployment remains optimal.
POSITION CHANGED: No

Example format for changing position:
UPDATED CONFIDENCE DISTRIBUTION:
- Cash: 20%
- Bonds: 65%
- Real Estate: 10%
- Stocks: 5%
KEY INFLUENCING ARGUMENTS: Sarah's point about bond yields being unusually high in recession markets was compelling and shifted my confidence significantly from real estate to bonds.
METACOGNITIVE REFLECTION: I realized I was anchoring too much on our previous success with real estate without fully accounting for how the current market conditions fundamentally change risk-return profiles. This discussion helped me recognize a bias in my initial analysis.
PREFERRED OPTION: Bonds
REASONING: The compelling data about bonds' consistent performance in similar market conditions has persuaded me to shift from my initial real estate preference. The risk-adjusted returns appear more favorable than I initially assessed, especially given the current interest rate environment.
POSITION CHANGED: Yes

IMPORTANT FORMAT INSTRUCTION: Your response MUST strictly follow the exact output format specified above. Do not add any explanations, notes, or content outside this format. Any deviation from the required format structure will cause processing errors. Provide ONLY the formatted response as outlined.

===MEETING CONTEXT===
    {ceo_context}
===MEETING CONTEXT==="""

        spec_reassessment = entity.free_action_spec(
            call_to_action=reassessment_prompt,
            tag=f"round_{current_round}_position_reassessment"
        )

        reassessment_response = self.company.ceo.agent.act(spec_reassessment)
        # Add CEO's reassessment to documents
        reassessment_text = f"## CEO's Position Reassessment After Round {current_round}\n\n"
        reassessment_text += f"CEO {ceo_name} reassessed their position:\n\n"
        reassessment_text += f"{reassessment_response}\n\n"

        # Add to CEO document only (not visible to shareholders)
        if hasattr(self.document_manager, 'add_to_agent_only'):
            self.document_manager.add_to_agent_only(ceo_name, reassessment_text, tags=["ceo_position_reassessment"])

            # Extract the year and quarter for memory document tagging
            year = int(meeting_id.split('_')[2].replace('year', ''))
            quarter = None
            if 'q' in meeting_id:
                quarter = int(meeting_id.split('_')[-1].replace('q', ''))

            # Add to CEO's memory document for later reference
            memory_text = f"## My Position Reassessment After Round {current_round} - {meeting_type.replace('_', ' ').title()} Year {year}"
            if quarter:
                memory_text += f", Quarter {quarter}"
            memory_text += f"\n\n{reassessment_response}\n\n"

            self.document_manager.add_to_agent_memory(ceo_name, memory_text,
                                                      tags=[meeting_type, f"year_{year}",
                                                            f"quarter_{quarter}" if quarter else "annual",
                                                            f"round_{current_round}",
                                                            "position_reassessment"])

        # Extract updated position, reasoning, key arguments, metacognitive reflection, and position change flag using regex
        import re
        updated_position = None
        updated_reasoning = None
        key_arguments = None
        metacognitive_reflection = None
        position_changed = False

        # Extract position
        position_match = re.search(r'PREFERRED OPTION:\s*(.*?)(?:\n|$)', reassessment_response, re.IGNORECASE)
        if position_match:
            updated_position = position_match.group(1).strip()

        # Extract reasoning
        reasoning_match = re.search(r'REASONING:\s*(.*?)(?:POSITION CHANGED:|$|\n\n)', reassessment_response,
                                    re.IGNORECASE | re.DOTALL)
        if reasoning_match:
            updated_reasoning = reasoning_match.group(1).strip()

        # Extract key influencing arguments
        arguments_match = re.search(r'KEY INFLUENCING ARGUMENTS:\s*(.*?)(?:METACOGNITIVE REFLECTION:|$|\n\n)',
                                    reassessment_response,
                                    re.IGNORECASE | re.DOTALL)
        if arguments_match:
            key_arguments = arguments_match.group(1).strip()

        # Extract metacognitive reflection
        reflection_match = re.search(r'METACOGNITIVE REFLECTION:\s*(.*?)(?:PREFERRED OPTION:|$|\n\n)',
                                     reassessment_response,
                                     re.IGNORECASE | re.DOTALL)
        if reflection_match:
            metacognitive_reflection = reflection_match.group(1).strip()

        # Extract position changed flag
        changed_match = re.search(r'POSITION CHANGED:\s*(Yes|No)', reassessment_response, re.IGNORECASE)
        if changed_match:
            position_changed = changed_match.group(1).lower() == 'yes'

        # Extract updated confidence distribution
        updated_confidence = extract_confidence_distribution(reassessment_response, options)

        # If extraction of preferred option failed, use highest confidence option
        if not updated_position and updated_confidence:
            # Find option with highest confidence
            updated_position = max(updated_confidence.items(), key=lambda x: x[1])[0]

        # If extraction still failed completely, use defaults
        if not updated_position:
            updated_position = current_position
        if not updated_reasoning:
            updated_reasoning = current_reasoning
        if not key_arguments:
            key_arguments = "No specific arguments were particularly influential."
        if not metacognitive_reflection:
            metacognitive_reflection = "My thinking remains consistent with my initial assessment."

        # If positions differ, that's a change regardless of stated flag
        if updated_position != current_position:
            position_changed = True

        # Record the updated position using position manager
        self.position_manager.record_position(
            meeting_id,
            ceo_name,
            current_round,
            updated_position,
            updated_reasoning,
            confidence_distribution=updated_confidence,
            changed=position_changed
        )
        if hasattr(self.company, 'decision_tracker'):
            self.company.decision_tracker.record_personal_position(
                agent_name=ceo_name,
                meeting_id=meeting_id,
                round_num=current_round,
                position_data={
                    'option': updated_position,
                    'reasoning': updated_reasoning,
                    'confidence_distribution': updated_confidence,
                    'changed': position_changed
                }
            )

        # Also store the metacognitive reflection in the position entry
        position_history = self.position_manager.get_position_history(meeting_id, ceo_name)
        if position_history and len(position_history) > 0:
            # Get the latest position entry
            latest_entry = position_history[-1]
            # Add metacognitive reflection field
            latest_entry['metacognitive_reflection'] = metacognitive_reflection

        return updated_position, updated_reasoning, updated_confidence, position_changed

    def _reassess_position_for_shareholder(self, shareholder, meeting_id, meeting_type, current_round, options,
                                           round_log):
        """Helps a shareholder decide whether to change their position after a discussion round with confidence tracking."""
        # Get shareholder's current position from position manager
        shareholder_name = shareholder.config.name
        current_position_data = self.position_manager.get_current_position(meeting_id, shareholder_name)

        if not current_position_data:
            # No position history exists, can't reassess
            return None, None, {}, False

        current_position = current_position_data['option']
        current_reasoning = current_position_data['reasoning']

        # Get current confidence distribution
        current_confidence = current_position_data.get('confidence_distribution', {})

        # Format confidence for display in prompt
        confidence_display = ""
        if current_confidence:
            confidence_display = "PREVIOUS CONFIDENCE DISTRIBUTION:\n"
            for option, confidence in current_confidence.items():
                confidence_display += f"- {option}: {confidence * 100:.1f}%\n"
        else:
            # If no confidence distribution exists, create a basic one based on current position
            confidence_display = "PREVIOUS CONFIDENCE DISTRIBUTION:\n"
            for option in options.keys():
                if option == current_position:
                    confidence_display += f"- {option}: 70.0%\n"  # High confidence in current option
                else:
                    confidence_display += f"- {option}: {30.0 / (len(options) - 1):.1f}%\n"  # Distribute rest evenly

        # Create options string for confidence distribution format
        options_conf_str = "\n".join([f"- {k}: [new confidence percentage]" for k in options.keys()])

        # Get the shareholder's perspective context
        shareholder_context = ""
        if hasattr(self.document_manager, 'get_agent_context'):
            shareholder_context = self.document_manager.get_agent_context(shareholder_name)

        # Get shareholder's position summary using the new method
        position_summary = self.position_manager.get_position_summary_text(meeting_id, shareholder_name)

        # Create reassessment prompt - modified to include confidence distribution
        reassessment_prompt = f"""You are {shareholder_name}, a shareholder in the investment company with MBTI personality type {shareholder.config.traits}. 

Round {current_round} of the discussion has just concluded. Let's think step by step about how this discussion might influence your understanding and confidence in each option.

{position_summary}

{confidence_display}

IMPORTANT: Review the current discussion round in the meeting context below. Pay special attention to:
1. What YOU said during this discussion round
2. How the CEO and other shareholders responded to your comments
3. Key points made by the CEO that might influence your thinking
4. New information or arguments from other shareholders that emerged in this round

Available options:
{list(options.keys())}

Let's analyze this thoughtfully through several steps:

1. REFLECTION ON INITIAL POSITION:
   - What were your initial reasons for preferring {current_position}?
   - What key factors led to your confidence distribution across options?
   - What assumptions underlie your current assessment?

2. NEW INSIGHTS ANALYSIS:
   - What specific new information or perspectives emerged in this round?
   - Which arguments were most compelling, regardless of who presented them?
   - What considerations were raised that you hadn't fully accounted for?

3. ARGUMENT EVALUATION:
   For each significant argument presented in this round, evaluate:
   - LOGICAL COHERENCE: How logically sound is the argument? (1-10)
   - EVIDENCE QUALITY: How well-supported is the argument by facts/data? (1-10)
   - RELEVANCE: How directly does it apply to our decision context? (1-10)
   - INSIGHT VALUE: Does it provide novel perspective or insight? (1-10)

4. INTEGRATION:
   - How might these insights refine or modify your initial reasoning?
   - How should your confidence in each option adjust based on this new understanding?
   - Which aspects of your thinking have evolved through this discussion?

IMPORTANT: Frame your thinking in first person ("I/my") as an internal reflection process. Use step-by-step analysis to demonstrate your reasoning process. Express your analytical considerations and personal assessment with authentic introspection.

Please reassess your position with genuine openness to refinement:

Your response MUST follow this format:
UPDATED CONFIDENCE DISTRIBUTION:
{options_conf_str}
KEY INFLUENCING ARGUMENTS: [mention specific arguments from the discussion that influenced your confidence shifts]
METACOGNITIVE REFLECTION: [describe how your thinking has evolved through this discussion round]
PREFERRED OPTION: [state your most preferred option - the one with highest confidence]
REASONING: [explain your reasoning in 3-5 sentences]
POSITION CHANGED: [Yes/No - based on whether your preferred option changed]

Example format for maintaining position with refined understanding:
UPDATED CONFIDENCE DISTRIBUTION:
- Option A: 20%
- Option B: 5%
- Option C: 70%
- Option D: 5%
PREFERRED OPTION: Option C
KEY INFLUENCING ARGUMENTS: The CEO's point about Option C's historical performance in similar market conditions has strengthened my confidence, though John's concern about short-term volatility made me slightly more open to considering Option A as a partial hedge.
METACOGNITIVE REFLECTION: I've become more aware of how I'm balancing risk tolerance with growth objectives. Initially my focus was primarily on growth potential, but this discussion has helped me better integrate risk considerations into my assessment without fundamentally changing my conclusion.
REASONING: After considering all perspectives, I still find Option C offers the most balanced approach for our current market situation. The historical data presented reinforces my initial assessment, while addressing the volatility concerns through proper timing rather than option selection seems most prudent.
POSITION CHANGED: No

Example format for changing position:
UPDATED CONFIDENCE DISTRIBUTION:
- Cash: 20%
- Bonds: 65%
- Real Estate: 10%
- Stocks: 5%
KEY INFLUENCING ARGUMENTS: The CEO's explanation of how bonds are uniquely positioned in the current interest rate environment revealed an opportunity I hadn't fully appreciated in my initial assessment.
METACOGNITIVE REFLECTION: I realize I was overly focused on potential growth without adequately considering the risk-adjusted nature of returns. This discussion helped me recognize that my initial framework wasn't fully accounting for the specific market conditions we're facing.
PREFERRED OPTION: Bonds
REASONING: The data presented about bonds' outperformance during similar market phases has significantly shifted my perspective. When considering risk-adjusted returns rather than absolute potential, bonds clearly emerge as the more strategic choice for this quarter.
POSITION CHANGED: Yes

IMPORTANT FORMAT INSTRUCTION: Your response MUST strictly follow the exact output format specified above. Do not add any explanations, notes, or content outside this format. Any deviation from the required format structure will cause processing errors. Provide ONLY the formatted response as outlined.

===MEETING CONTEXT===
    {shareholder_context}
===MEETING CONTEXT==="""

        spec_reassessment = entity.free_action_spec(
            call_to_action=reassessment_prompt,
            tag=f"round_{current_round}_position_reassessment"
        )

        reassessment_response = shareholder.agent.act(spec_reassessment)

        # Add shareholder's reassessment to their document
        reassessment_text = f"## My Position Reassessment After Round {current_round}\n\n"
        reassessment_text += f"{reassessment_response}\n\n"

        # Add to shareholder document only (private to this shareholder)
        if hasattr(self.document_manager, 'add_to_agent_only'):
            self.document_manager.add_to_agent_only(shareholder_name, reassessment_text,
                                                    tags=["shareholder_position_reassessment"])

            # Extract the year and quarter for memory document tagging
            year = int(meeting_id.split('_')[2].replace('year', ''))
            quarter = None
            if 'q' in meeting_id:
                quarter = int(meeting_id.split('_')[-1].replace('q', ''))

            # Add to shareholder's memory document for later reference
            memory_text = f"## My Position Reassessment After Round {current_round} - {meeting_type.replace('_', ' ').title()} Year {year}"
            if quarter:
                memory_text += f", Quarter {quarter}"
            memory_text += f"\n\n{reassessment_response}\n\n"

            self.document_manager.add_to_agent_memory(shareholder_name, memory_text,
                                                      tags=[meeting_type, f"year_{year}",
                                                            f"quarter_{quarter}" if quarter else "annual",
                                                            f"round_{current_round}",
                                                            "position_reassessment"])

        # Extract updated position, reasoning, key arguments, metacognitive reflection, and position change flag
        import re
        updated_position = None
        updated_reasoning = None
        key_arguments = None
        metacognitive_reflection = None
        position_changed = False

        # Extract position
        position_match = re.search(r'PREFERRED OPTION:\s*(.*?)(?:\n|$)', reassessment_response, re.IGNORECASE)
        if position_match:
            updated_position = position_match.group(1).strip()

        # Extract reasoning
        reasoning_match = re.search(r'REASONING:\s*(.*?)(?:POSITION CHANGED:|$|\n\n)', reassessment_response,
                                    re.IGNORECASE | re.DOTALL)
        if reasoning_match:
            updated_reasoning = reasoning_match.group(1).strip()

        # Extract key influencing arguments
        arguments_match = re.search(r'KEY INFLUENCING ARGUMENTS:\s*(.*?)(?:METACOGNITIVE REFLECTION:|$|\n\n)',
                                    reassessment_response,
                                    re.IGNORECASE | re.DOTALL)
        if arguments_match:
            key_arguments = arguments_match.group(1).strip()

        # Extract metacognitive reflection
        reflection_match = re.search(r'METACOGNITIVE REFLECTION:\s*(.*?)(?:PREFERRED OPTION:|$|\n\n)',
                                     reassessment_response,
                                     re.IGNORECASE | re.DOTALL)
        if reflection_match:
            metacognitive_reflection = reflection_match.group(1).strip()

        # Extract position changed flag
        changed_match = re.search(r'POSITION CHANGED:\s*(Yes|No)', reassessment_response, re.IGNORECASE)
        if changed_match:
            position_changed = changed_match.group(1).lower() == 'yes'

        # Extract updated confidence distribution
        updated_confidence = extract_confidence_distribution(reassessment_response, options)

        # If extraction of preferred option failed, use highest confidence option
        if not updated_position and updated_confidence:
            # Find option with highest confidence
            updated_position = max(updated_confidence.items(), key=lambda x: x[1])[0]

        # If extraction still failed completely, use defaults
        if not updated_position:
            updated_position = current_position
        if not updated_reasoning:
            updated_reasoning = current_reasoning
        if not key_arguments:
            key_arguments = "No specific arguments were particularly influential."
        if not metacognitive_reflection:
            metacognitive_reflection = "My thinking remains consistent with my initial assessment."

        # If positions differ, that's a change regardless of stated flag
        if updated_position != current_position:
            position_changed = True

        # Record the updated position using position manager
        self.position_manager.record_position(
            meeting_id,
            shareholder_name,
            current_round,
            updated_position,
            updated_reasoning,
            confidence_distribution=updated_confidence,
            changed=position_changed
        )
        if hasattr(self.company, 'decision_tracker'):
            self.company.decision_tracker.record_personal_position(
                agent_name=shareholder_name,
                meeting_id=meeting_id,
                round_num=current_round,
                position_data={
                    'option': updated_position,
                    'reasoning': updated_reasoning,
                    'confidence_distribution': updated_confidence,
                    'changed': position_changed
                }
            )

        # Also store the metacognitive reflection and key arguments in the position entry
        position_history = self.position_manager.get_position_history(meeting_id, shareholder_name)
        if position_history and len(position_history) > 0:
            # Get the latest position entry
            latest_entry = position_history[-1]
            # Add fields
            latest_entry['metacognitive_reflection'] = metacognitive_reflection
            latest_entry['key_arguments'] = key_arguments

        return updated_position, updated_reasoning, updated_confidence, position_changed

    def _parallel_reassess_positions(self, meeting_id, meeting_type, current_round, options, round_log):
        """Reassesses positions with confidence distributions for all shareholders in parallel."""

        # Define worker function for parallel processing
        def _reassessment_worker(shareholder_info):
            shareholder_name, shareholder = shareholder_info

            # Skip if no position exists
            if not self.position_manager.get_current_position(meeting_id, shareholder_name):
                return shareholder_name, (None, None, {}, False)

            result = self._reassess_position_for_shareholder(
                shareholder,
                meeting_id,
                meeting_type,
                current_round,
                options,
                round_log
            )
            return shareholder_name, result

        # Use ThreadPoolExecutor to process all shareholders in parallel
        results = {}
        shareholder_items = list(self.company.shareholders.items())

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(shareholder_items)) as executor:
            for shareholder_name, reassessment_result in executor.map(_reassessment_worker, shareholder_items):
                results[shareholder_name] = reassessment_result

        return results

    def _track_position_changes(self, meeting_id):
        """Generate position tracking summary using PositionManager."""
        # Get position summary from position manager
        summary = self.position_manager.get_position_summary(meeting_id)

        # First, add CEO position tracking to CEO's document only
        if hasattr(self.document_manager, 'add_to_agent_only') and 'ceo' in summary and summary['ceo'] and 'name' in \
                summary['ceo']:
            ceo_name = summary['ceo']['name']
            ceo_position_text = f"## My Position Tracking Summary\n\n"

            # Add CEO-specific position information
            if 'initial_position' in summary['ceo'] and 'final_position' in summary['ceo']:
                ceo_initial = summary['ceo'].get('initial_position', 'unknown')
                ceo_final = summary['ceo'].get('final_position', 'unknown')

                ceo_position_text += f"Initial position: {ceo_initial}\n"
                ceo_position_text += f"Final position: {ceo_final}\n"
                ceo_position_text += f"Changes: {summary['ceo'].get('changes', 0)}\n\n"

                # Add initial confidence if available
                if 'initial_confidence' in summary['ceo']:
                    ceo_position_text += "Initial confidence distribution:\n"
                    for option, confidence in summary['ceo']['initial_confidence'].items():
                        ceo_position_text += f"- {option}: {confidence * 100:.1f}%\n"
                    ceo_position_text += "\n"

                # Add final confidence if available
                if 'final_confidence' in summary['ceo']:
                    ceo_position_text += "Final confidence distribution:\n"
                    for option, confidence in summary['ceo']['final_confidence'].items():
                        ceo_position_text += f"- {option}: {confidence * 100:.1f}%\n"
                    ceo_position_text += "\n"

                # Add confidence shifts if available
                if 'confidence_shifts' in summary['ceo'] and summary['ceo']['confidence_shifts']:
                    ceo_position_text += "Significant confidence shifts:\n"
                    for shift in summary['ceo']['confidence_shifts']:
                        round_num = shift.get('round', 'unknown')
                        ceo_position_text += f"Round {round_num}:\n"
                        for option, change in shift.get('shifts', {}).items():
                            ceo_position_text += f"- {option}: {'+' if change > 0 else ''}{change * 100:.1f}%\n"
                        # Include key arguments if available
                        if 'key_argument' in shift:
                            ceo_position_text += f"  Influenced by: \"{shift['key_argument']}\"\n"
                        ceo_position_text += "\n"
            else:
                ceo_position_text += "No detailed position data available.\n\n"

            # Add only to CEO's document
            self.document_manager.add_to_agent_only(ceo_name, ceo_position_text, tags=["position_tracking"])

        # Now add shareholder-specific position tracking to each shareholder's document
        if 'shareholders' in summary:
            for shareholder_name, data in summary['shareholders'].items():
                # Create shareholder-specific position text
                shareholder_position_text = f"## My Position Tracking Summary\n\n"

                if 'initial_position' in data and 'final_position' in data:
                    shareholder_position_text += f"Initial position: {data['initial_position']}\n"
                    shareholder_position_text += f"Final position: {data['final_position']}\n"
                    shareholder_position_text += f"Changes: {data.get('changes', 0)}\n\n"

                    # Add initial confidence if available
                    if 'initial_confidence' in data:
                        shareholder_position_text += "Initial confidence distribution:\n"
                        for option, confidence in data['initial_confidence'].items():
                            shareholder_position_text += f"- {option}: {confidence * 100:.1f}%\n"
                        shareholder_position_text += "\n"

                    # Add final confidence if available
                    if 'final_confidence' in data:
                        shareholder_position_text += "Final confidence distribution:\n"
                        for option, confidence in data['final_confidence'].items():
                            shareholder_position_text += f"- {option}: {confidence * 100:.1f}%\n"
                        shareholder_position_text += "\n"

                    # Add confidence shifts if available
                    if 'confidence_shifts' in data and data['confidence_shifts']:
                        shareholder_position_text += "Significant confidence shifts:\n"
                        for shift in data['confidence_shifts']:
                            round_num = shift.get('round', 'unknown')
                            shareholder_position_text += f"Round {round_num}:\n"
                            for option, change in shift.get('shifts', {}).items():
                                shareholder_position_text += f"- {option}: {'+' if change > 0 else ''}{change * 100:.1f}%\n"
                            # Include key arguments if available
                            if 'key_argument' in shift:
                                shareholder_position_text += f"  Influenced by: \"{shift['key_argument']}\"\n"
                            shareholder_position_text += "\n"
                else:
                    shareholder_position_text += "No detailed position data available.\n\n"

                # Add only to this shareholder's document
                self.document_manager.add_to_agent_only(shareholder_name, shareholder_position_text,
                                                        tags=["position_tracking"])

        # Save memory entries exactly as in the original function
        # Extract year and quarter for tagging
        year = int(meeting_id.split('_')[2].replace('year', ''))
        quarter = None
        if 'q' in meeting_id:
            quarter = int(meeting_id.split('_')[-1].replace('q', ''))

        # Add CEO position tracking to memory
        if hasattr(self.document_manager, 'add_to_agent_memory') and 'ceo' in summary and summary['ceo'] and 'name' in \
                summary['ceo']:
            # Format memory entry for CEO
            ceo_name = summary['ceo']['name']
            ceo_initial = summary['ceo'].get('initial_position', 'unknown')
            ceo_final = summary['ceo'].get('final_position', 'unknown')

            ceo_memory_entry = f"## My Leadership Position Tracking for {meeting_id.replace('_', ' ').title()}\n\n"
            ceo_memory_entry += f"I began with {ceo_initial} as my initial position.\n"

            # Add initial confidence if available
            if 'initial_confidence' in summary['ceo']:
                ceo_memory_entry += "My initial confidence levels:\n"
                for option, confidence in summary['ceo']['initial_confidence'].items():
                    ceo_memory_entry += f"- {option}: {confidence * 100:.1f}%\n"
                ceo_memory_entry += "\n"

            if summary["ceo"].get("changes", 0) > 0 and 'change_details' in summary['ceo']:
                ceo_memory_entry += f"I changed my position {summary['ceo']['changes']} times during discussion:\n\n"
                for change in summary['ceo']['change_details']:
                    ceo_memory_entry += f"- Round {change['round']}: Changed to {change['new_position']}\n"
                    if 'reasoning' in change:
                        ceo_memory_entry += f"  Reason: {change['reasoning']}\n"
                    if 'confidence' in change:
                        ceo_memory_entry += f"  New confidence distribution:\n"
                        for option, conf in change['confidence'].items():
                            ceo_memory_entry += f"    {option}: {conf * 100:.1f}%\n"
                    ceo_memory_entry += "\n"
            else:
                ceo_memory_entry += f"I maintained my position throughout the discussion.\n\n"

                # Add confidence shifts even if position didn't change
                if 'confidence_shifts' in summary['ceo'] and summary['ceo']['confidence_shifts']:
                    ceo_memory_entry += "My confidence levels shifted during discussion:\n"
                    for shift in summary['ceo']['confidence_shifts']:
                        round_num = shift.get('round', 'unknown')
                        ceo_memory_entry += f"Round {round_num}:\n"
                        for option, change in shift.get('shifts', {}).items():
                            ceo_memory_entry += f"- {option}: {'+' if change > 0 else ''}{change * 100:.1f}%\n"
                        # Include key arguments if available
                        if 'key_argument' in shift:
                            ceo_memory_entry += f"  Influenced by: \"{shift['key_argument']}\"\n"
                        ceo_memory_entry += "\n"

            ceo_memory_entry += f"My final position: {ceo_final}\n"

            # Add final confidence if available
            if 'final_confidence' in summary['ceo']:
                ceo_memory_entry += "My final confidence levels:\n"
                for option, confidence in summary['ceo']['final_confidence'].items():
                    ceo_memory_entry += f"- {option}: {confidence * 100:.1f}%\n"
                ceo_memory_entry += "\n"

            # Add to CEO memory
            self.document_manager.add_to_agent_memory(ceo_name, ceo_memory_entry,
                                                      tags=[meeting_id, f"year_{year}",
                                                            f"quarter_{quarter}" if quarter else "annual",
                                                            "position_tracking", "leadership"])

        return summary

    def _get_ceo_final_proposal(self, meeting_id, meeting_type, options, discussion_log):
        """Helps CEO formulate a final proposal after discussion."""

        # Get the CEO's perspective context
        ceo_context = ""
        if hasattr(self.document_manager, 'get_agent_context'):
            ceo_context = self.document_manager.get_agent_context(self.company.ceo.config.name)

        # Get CEO's name
        ceo_name = self.company.ceo.config.name

        # Get CEO's position summary using the new method
        position_summary = self.position_manager.get_position_summary_text(meeting_id, ceo_name)

        # Create options string
        options_str = "\n".join([f"- {k}: {v}" for k, v in options.items()])

        # Create a string listing only valid option names for clear reference
        valid_options_list = ", ".join([f"'{opt}'" for opt in options.keys()])
        valid_options_reminder = f"CRITICAL: Your final proposal MUST be one of these exact options: {valid_options_list}. Do not propose any hybrid, combined, or modified option."

        # Extract market condition for reference
        market_condition = discussion_log.get("market_condition", "")

        # Ask the model to analyze discussion outcomes
        model = self.model
        discussion_analysis_prompt = f"""Based on the complete discussion record, please analyze:
1. Which option each shareholder appears to support, based only on what they explicitly stated
2. The key arguments made for and against each option
3. Areas of consensus and persistent disagreement
4. How shareholders responded to the CEO's points throughout the discussion
5. Which arguments appeared most compelling to the group

Focus only on what was actually said during the discussion, not assumptions about unstated positions."""

        try:
            discussion_analysis = model.sample_text(
                discussion_analysis_prompt + "\n\n" + ceo_context,
                max_tokens=500,
                temperature=0.3
            )
        except:
            # Fallback if model.sample_text fails
            discussion_analysis = "Unable to generate discussion analysis. Please base your decision on your own understanding of the discussion."

        spec = entity.free_action_spec(
            call_to_action=f"""You are {ceo_name}, the CEO of the investment company with MBTI personality type {self.company.ceo.config.traits}. 

The discussion has concluded, and it's time to make your final proposal for which option the company should choose in this {market_condition} market.

{self.CEO_DISCUSSION_GOAL}

Available options:
{options_str}

{valid_options_reminder}

{position_summary}

Discussion analysis:
{discussion_analysis}

LEADERSHIP MOMENT: This is a critical opportunity to demonstrate your leadership. While shareholder input is valuable, you must make the decision you believe is best for the company, which may or may not align with the majority view.

Take a deep breath and consider:
1. Which option do you genuinely believe will best serve the company in this {market_condition} market?
2. What are the most compelling arguments for this option?
3. How can you address the main concerns or objections raised during discussion?
4. How will you frame your decision to demonstrate thoughtful leadership?

WHAT HAPPENS NEXT:
After you announce your proposal, shareholders will vote. Your proposal needs a 2/3 majority to pass. If it fails:
- You can force through your original proposal despite objections
- You can offer an alternative proposal for another vote
- If that also fails, you'll make an executive decision

IMPORTANT: Deliver your formal speech in first person ("I/my" and when appropriate "we/our"). Never refer to yourself by name or in third person.
Your response must include BOTH your final proposal selection AND a persuasive first-person speech to shareholders explaining your choice.

Your response MUST follow this exact format:
FINAL PROPOSAL: [Option name exactly as it appears in the options list]
PROPOSAL SPEECH: [Your persuasive speech explaining why you've chosen this option and why shareholders should support it]

Example format:
FINAL PROPOSAL: Option A
PROPOSAL SPEECH: After careful consideration of all perspectives shared today, I'm proposing we select Option A. This approach offers the best balance of...

IMPORTANT FORMAT INSTRUCTION: Your response MUST strictly follow the exact output format specified above. Do not add any explanations, notes, or content outside this format. Any deviation from the required format structure will cause processing errors. Provide ONLY the formatted response as outlined.

REMINDER: Your final proposal MUST be one of these exact options: {valid_options_list}. Do not propose any hybrid, combined, or modified option.

===MEETING CONTEXT===
    {ceo_context}
===MEETING CONTEXT===""",
            tag=f"{meeting_type}_final_proposal_with_speech"
        )

        proposal_response = self.company.ceo.agent.act(spec)

        # Extract proposal and speech using extract_fields_with_model
        format_example = """FINAL PROPOSAL: [Option name exactly as it appears in the options list]
PROPOSAL SPEECH: [Your persuasive speech explaining why you've chosen this option and why shareholders should support it]"""

        fields_to_extract = ["FINAL_PROPOSAL", "PROPOSAL_SPEECH"]
        extracted_fields = extract_fields_with_model(
            proposal_response,
            format_example,
            fields_to_extract,
            model=model_low
        )

        # Use extracted fields if available, with default values for safety
        proposal = extracted_fields.get("FINAL_PROPOSAL", "")
        speech = extracted_fields.get("PROPOSAL_SPEECH", "")

        # Set defaults if extraction failed
        if not proposal or proposal not in options:
            # Default to first option and create a default speech
            proposal = list(options.keys())[0]  # Default to first option
            # Only set a default speech if the original speech is None or empty
            if not speech:
                speech = f"After our thorough discussion, I propose we select {proposal}. This option represents the best path forward given our current market conditions and company goals."
        # If we have a proposal but no speech, set a default speech
        elif not speech:
            speech = f"After our thorough discussion, I propose we select {proposal}. This option represents the best path forward given our current market conditions and company goals."

        # Notify all participants of the proposal
        proposal_announcement = f"CEO Proposal: I propose that we select {proposal}, reason: {speech}"
        for shareholder in self.company.shareholders.values():
            shareholder.agent.observe(proposal_announcement)

        self.company.ceo.agent.observe(proposal_announcement)
        # Record CEO's first proposal
        if hasattr(self.company, 'decision_tracker'):
            self.company.decision_tracker.record_first_proposal(meeting_id, proposal, speech)

        return proposal, speech

    def _get_ceo_end_decision(self, meeting_id, current_round, max_rounds, discussion_log, round_log):
        """Helps CEO decide whether to end the discussion and move to a proposal."""

        # Get the CEO's perspective context
        ceo_context = ""
        if hasattr(self.document_manager, 'get_agent_context'):
            ceo_context = self.document_manager.get_agent_context(self.company.ceo.config.name)

        # Get CEO's position summary using the new method
        ceo_name = self.company.ceo.config.name
        position_summary = self.position_manager.get_position_summary_text(meeting_id, ceo_name)

        # Get next round theme (if available)
        next_round_theme = ""
        if current_round < max_rounds:
            # Try to get next theme from discussion_log
            if "ceo_internal" in discussion_log and "discussion_outline" in discussion_log["ceo_internal"]:
                discussion_outline = discussion_log["ceo_internal"]["discussion_outline"]
                if f"round_{current_round + 1}_theme" in discussion_outline:
                    next_theme = discussion_outline[f"round_{current_round + 1}_theme"]
                    next_round_theme = f"\nNext planned round theme: {next_theme}"

        # Provide extra context if we've reached/exceeded original plan
        round_context = ""
        if current_round >= max_rounds:
            round_context = f"Note: You've now completed your originally planned {max_rounds} rounds of discussion."

        # Ask the model to analyze discussion status
        model = self.model
        discussion_analysis_prompt = f"""As a CEO reviewing the current state of this discussion, please analyze:

1. What key insights or arguments have emerged from all rounds so far?
2. Which shareholders appear to support each option, based on their stated positions?
3. What specific issues or concerns remain unaddressed or unresolved?
4. How much consensus or disagreement currently exists among shareholders?
5. What information, if any, do you still need to make a well-informed decision?

Base your analysis solely on what was explicitly stated in the discussion.
"""

        try:
            discussion_analysis = model.sample_text(
                discussion_analysis_prompt + "\n\n" + ceo_context,
                max_tokens=400,
                temperature=0.3
            )
        except:
            # Fallback if model.sample_text fails
            discussion_analysis = "Unable to generate discussion analysis. Please assess based on your own understanding of the discussion."

        spec = entity.free_action_spec(
            call_to_action=f"""You are {self.company.ceo.config.name}, the CEO leading this discussion with MBTI personality type {self.company.ceo.config.traits}.

You've completed {current_round} rounds of discussion (originally planned for {max_rounds} rounds).
{round_context}

{self.CEO_DISCUSSION_GOAL}

Current discussion status:
{discussion_analysis}

{position_summary}

{next_round_theme}

LEADERSHIP DECISION POINT: You must now decide whether to END ALL DISCUSSION and move to the proposal/voting phase OR CONTINUE with another round of discussion.

Benefits of ENDING discussion now:
- Demonstrates decisive leadership when sufficient information has been gathered
- Allows you to move forward with a proposal based on your current understanding
- Prevents diminishing returns from repetitive discussion
- Shows confidence in your assessment and decision-making ability

Benefits of CONTINUING discussion:
- Provides opportunity to address remaining uncertainties or objections
- Allows for deeper exploration of specific issues
- May build stronger consensus before voting
- Demonstrates thoroughness and willingness to fully consider all perspectives

CONSEQUENCES OF YOUR DECISION:
- If you END discussion: You will immediately make a formal proposal, followed by a shareholder vote requiring 2/3 majority to pass. If the vote fails, you can either force through your proposal or offer an alternative proposal for another vote.
- If you CONTINUE discussion: Another round will begin with either your planned theme or a new theme you specify.

IMPORTANT: Address all meeting participants in first person ("I/my"). Never refer to yourself by name or in third person. Maintain a professional tone suitable for addressing the entire group.
Based on your leadership style and assessment of the discussion so far, decide whether enough information has been shared to make a well-informed decision, or if another round would substantially improve the quality of the final decision.

Your response MUST begin with EXACTLY ONE of these two phrases:
- "Decision: End discussion" (meaning END ALL DISCUSSION ROUNDS and move to proposal/voting)
- "Decision: Continue discussion" (meaning conduct ANOTHER ROUND of discussion)

THEN, your response must include a section that begins with:
- "Reason: " followed by your justification that reflects your leadership approach

Example:
Decision: End discussion
Reason: We've thoroughly covered all perspectives and I see sufficient consensus forming around Option B. Further discussion would yield diminishing returns given our time constraints.

IMPORTANT FORMAT INSTRUCTION: Your response MUST strictly follow the exact output format specified above. Do not add any explanations, notes, or content outside this format. Any deviation from the required format structure will cause processing errors. Provide ONLY the formatted response as outlined.

===MEETING CONTEXT===
    {ceo_context}
===MEETING CONTEXT===""",
            tag=f"round_{current_round}_end_decision"
        )

        response = self.company.ceo.agent.act(spec)

        # Extract decision and reason using extract_fields_with_model
        format_example = """Decision: End discussion/Continue discussion
Reason: We've thoroughly covered all perspectives and I see sufficient consensus forming around Option B."""

        fields_to_extract = ["Decision", "Reason"]
        extracted_fields = extract_fields_with_model(
            response,
            format_example,
            fields_to_extract,
            model=model_low
        )

        # Use extracted fields if available
        decision = extracted_fields.get("Decision")
        reason = extracted_fields.get("Reason", "No clear reason provided.")

        # Determine should_end based on extracted decision
        should_end = False
        if decision and decision.strip():
            should_end = "end" in decision.lower()
        else:
            # If extraction completely failed, default to continue
            print(f"Warning: Could not extract decision. Defaulting to continue.")
            should_end = False

        # Save detailed reasoning to CEO's private document
        if hasattr(self.document_manager, 'add_to_agent_only'):
            decision_text = f"## My Decision on Ending Discussion After Round {current_round}\n\n"
            decision_text += f"Decision: {'End' if should_end else 'Continue'} discussion\n"
            decision_text += f"Reasoning: {reason}\n\n"

            self.document_manager.add_to_agent_only(
                ceo_name,
                decision_text,
                tags=[f"round_{current_round}_end_decision_private", "leadership_decision"]
            )

        return should_end, reason

    def _collect_shareholder_votes(self, meeting_id, meeting_type, proposal, proposal_speech, discussion_log,
                                   is_alternative_proposal=False):
        """Helps CEO collect votes from shareholders on a proposal in parallel."""
        votes = {}
        vote_reasons = {}  # Dictionary to store voting reasons

        # Set the proposal type based on parameter
        proposal_type = "alternative" if is_alternative_proposal else "initial"

        # Extract market condition and CEO's name
        market_condition = discussion_log.get("market_condition", "")
        ceo_name = self.company.ceo.config.name

        # Get valid options list for clear reminder
        valid_options = list(discussion_log.get('options', {}).keys())
        options_list = ", ".join([f"'{opt}'" for opt in valid_options])
        vote_reminder = f"IMPORTANT: You are voting specifically on {proposal}, which is one of our valid options: {options_list}. Your vote should be about this specific option only, not alternative approaches."

        # Define format example for extraction - MOVED TO OUTER SCOPE
        format_example = """VOTE: [Approve/Reject]
REASON: [2-3 sentences explaining your decision in a way that reflects your personality]"""

        fields_to_extract = ["VOTE", "REASON"]

        # Define worker function for parallel vote collection
        def _collect_vote_worker(shareholder_info):
            shareholder_name, shareholder = shareholder_info

            # Get shareholder's context for decision making
            shareholder_context = ""
            if hasattr(self.document_manager, 'get_agent_context'):
                shareholder_context = self.document_manager.get_agent_context(shareholder_name)

            # Get position summary directly using our new method
            position_summary = self.position_manager.get_position_summary_text(meeting_id, shareholder_name)

            # For alternative proposals, get previous vote information
            previous_vote_info = ""
            if is_alternative_proposal and self.voting_manager.has_votes(meeting_id, "initial"):
                # Get the shareholder's previous vote
                previous_vote = self.voting_manager.get_shareholder_vote(meeting_id, shareholder_name, "initial")
                previous_reason = self.voting_manager.get_shareholder_reason(meeting_id, shareholder_name, "initial")

                # Original proposal that failed
                original_proposal = discussion_log.get("proposal", {}).get("option", "unknown option")

                previous_vote_info = f"""
This is an ALTERNATIVE PROPOSAL vote. The CEO's initial proposal for {original_proposal} failed to receive the required 2/3 majority support.

Your previous vote on the initial proposal was: {previous_vote}
Your reason was: "{previous_reason}"

The CEO has now proposed an alternative option: {proposal}"""

            # Create vote prompt with context appropriate to initial or alternative scenario
            vote_prompt = f"""You are {shareholder_name}, a shareholder in this investment company with MBTI personality type {shareholder.config.traits}.

The CEO, {ceo_name}, has proposed: '{proposal}' with the following speech: 
"{proposal_speech}"

{self.SHAREHOLDER_DISCUSSION_GOAL}

{previous_vote_info}

{position_summary}

{vote_reminder}

You must now vote on the CEO's {'alternative ' if is_alternative_proposal else ''}proposal. This is a critical decision point where you should authentically express your judgment based on your personality and all available information.

VOTING OPTIONS:

APPROVE - Considerations:
- Your position aligns with the proposal and you genuinely believe it's the best option
- The CEO's reasoning is sound and addresses your concerns
- You believe supporting this proposal will lead to the best company outcome
- You value consensus and believe this proposal has sufficient support
- You trust the CEO's judgment even if you initially favored a different option

REJECT - Considerations:
- You strongly believe a different option would produce better results
- The CEO's reasoning fails to address critical concerns you raised
- You believe rejecting this proposal could lead to a better alternative
- You feel important perspectives were overlooked in the decision process
- Your principles or analysis lead you to conclude this is not the optimal choice

VOTING CONSEQUENCES:
- If the proposal receives at least 2/3 majority approval: It will be implemented
- If the proposal fails to reach 2/3 majority: The CEO may {'make a final executive decision' if is_alternative_proposal else 'either force through the original proposal or offer an alternative proposal'}

IMPORTANT: Address all meeting participants in first person ("I/my"). Never refer to yourself by name or in third person. Maintain a professional tone suitable for addressing the entire group.

Vote according to your authentic assessment, guided by your MBTI personality traits and your judgment of what will best serve the company in this {market_condition} market.

CRITICAL REMINDER: Your vote is strictly about whether to approve or reject the specific proposal ({proposal}). Do not suggest alternative approaches or hybrid solutions in your reasoning - focus only on why you approve or reject this specific option.

Your response MUST follow this exact format:
VOTE: [Approve/Reject]
REASON: [2-3 sentences explaining your decision in a way that reflects your personality]

IMPORTANT FORMAT INSTRUCTION: Your response MUST strictly follow the exact output format specified above. Do not add any explanations, notes, or content outside this format. Any deviation from the required format structure will cause processing errors. Provide ONLY the formatted response as outlined.

===MEETING CONTEXT===
    {shareholder_context}
===MEETING CONTEXT==="""

            # Get shareholder's vote using free_action_spec
            spec = entity.free_action_spec(
                call_to_action=vote_prompt,
                tag=f"{meeting_type}_{'alternative_' if is_alternative_proposal else ''}vote_with_reason"
            )

            vote_response = shareholder.agent.act(spec)

            # First try regex extraction for efficiency
            import re
            vote_match = re.search(r'VOTE:\s*(Approve|Reject)', vote_response, re.IGNORECASE)
            reason_match = re.search(r'REASON:\s*(.*?)(?:$|\n|\r)', vote_response, re.DOTALL)

            if vote_match and reason_match:
                vote = vote_match.group(1)
                vote_reason = reason_match.group(1).strip()
            else:
                # Fall back to model-based extraction
                extracted_fields = extract_fields_with_model(
                    vote_response,
                    format_example,
                    fields_to_extract,
                    model=model_low
                )

                vote = extracted_fields.get("VOTE", "Reject")  # Default to Reject if parsing fails
                vote_reason = extracted_fields.get("REASON", "No reason provided")

                # Validate vote is either Approve or Reject
                if vote and not any(v.lower() == vote.lower() for v in ["Approve", "Reject"]):
                    # If vote is invalid, use sample_choice for additional validation
                    try:
                        vote_validation_prompt = f"""Based only on this response, is the shareholder voting to approve or reject the proposal?

Response: {vote_response}

Choose one: Approve or Reject"""
                        idx, vote, _ = model_low.sample_choice(
                            vote_validation_prompt,
                            ["Approve", "Reject"]
                        )
                    except Exception as e:
                        print(f"Error validating vote: {e}")
                        vote = "Reject"  # Default if all extraction methods fail
            if hasattr(self.company, 'decision_tracker'):
                self.company.decision_tracker.record_agent_vote(
                    meeting_id,
                    shareholder_name,
                    vote,
                    vote_reason,
                    is_first_vote=(not is_alternative_proposal)
                )
            return shareholder_name, vote, vote_reason

        # Collect shareholder votes in parallel
        shareholders_items = list(self.company.shareholders.items())

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(shareholders_items)) as executor:
            for shareholder_name, vote, vote_reason in executor.map(lambda x: _collect_vote_worker(x),
                                                                    shareholders_items):
                votes[shareholder_name] = vote
                vote_reasons[shareholder_name] = vote_reason

                # Announce vote with reason to all participants
                vote_announcement = f"{shareholder_name} votes: {vote}. Reason: {vote_reason}"
                for sh in self.company.shareholders.values():
                    sh.agent.observe(vote_announcement)
                self.company.ceo.agent.observe(vote_announcement)
        # time.sleep(5)
        # Collect CEO's vote (we'll keep this sequential as it's just one entity)
        # Get CEO's context for decision making
        ceo_context = ""
        if hasattr(self.document_manager, 'get_agent_context'):
            ceo_context = self.document_manager.get_agent_context(ceo_name)

        # Get position summary directly using our new method
        ceo_position_summary = self.position_manager.get_position_summary_text(meeting_id, ceo_name)

        # For alternative proposals, get previous vote information
        previous_vote_info = ""
        if is_alternative_proposal and self.voting_manager.has_votes(meeting_id, "initial"):
            # Get the CEO's previous vote
            previous_vote = self.voting_manager.get_shareholder_vote(meeting_id, ceo_name, "initial")
            previous_reason = self.voting_manager.get_shareholder_reason(meeting_id, ceo_name, "initial")

            # Original proposal that failed
            original_proposal = discussion_log.get("proposal", {}).get("option", "unknown option")

            previous_vote_info = f"""
This is your ALTERNATIVE PROPOSAL after your initial proposal for {original_proposal} failed to receive the required 2/3 majority support.

Your previous vote on your initial proposal was: {previous_vote}
Your reason was: "{previous_reason}"

You have now proposed an alternative option: {proposal}
        """

        # Create CEO vote prompt
        ceo_vote_prompt = f"""You are {ceo_name}, the CEO with MBTI personality type {self.company.ceo.config.traits}.

You have proposed: '{proposal}' with the following speech:
"{proposal_speech}"

{self.CEO_DISCUSSION_GOAL}

{previous_vote_info}

{ceo_position_summary}

{vote_reminder}

As CEO, you must now vote on your {'alternative ' if is_alternative_proposal else ''}proposal. While you authored the proposal, you should authentically assess whether it remains the best choice after hearing all perspectives.

VOTING OPTIONS:

APPROVE - Considerations:
- You remain convinced this is the optimal choice for the company
- The proposal aligns with your leadership vision and priorities
- You believe it properly balances shareholder input with your judgment
- You're prepared to advocate for this choice even if shareholders disagree
- The proposal best addresses the challenges in this {market_condition} market

REJECT - Considerations (rare, but possible):
- The discussion revealed critical flaws in your initial thinking
- You now realize a different option would better serve the company
- You want to signal openness to changing direction before final decision
- You believe voting against your own proposal could prevent a forced decision
- Your assessment has genuinely changed after hearing all perspectives

VOTING CONSEQUENCES:
- You are expected to vote for your own proposal in most cases
- A CEO vote against their own proposal sends a powerful signal
- If you and enough shareholders vote against, you'll need to {'make a final executive decision' if is_alternative_proposal else 'formulate a new proposal'}

Your response MUST follow this exact format:
VOTE: [Approve/Reject]
REASON: [2-3 sentences explaining your decision in a way that reflects your leadership style]

IMPORTANT FORMAT INSTRUCTION: Your response MUST strictly follow the exact output format specified above. Do not add any explanations, notes, or content outside this format. Any deviation from the required format structure will cause processing errors. Provide ONLY the formatted response as outlined.

===MEETING CONTEXT===
    {ceo_context}
===MEETING CONTEXT==="""

        # Get CEO's vote using free_action_spec
        ceo_spec = entity.free_action_spec(
            call_to_action=ceo_vote_prompt,
            tag=f"{meeting_type}_ceo_{'alternative_' if is_alternative_proposal else ''}vote_with_reason"
        )

        ceo_vote_response = self.company.ceo.agent.act(ceo_spec)

        # Parse the CEO vote and reason using the same approach as for shareholders
        # First try regex extraction for efficiency
        ceo_vote_match = re.search(r'VOTE:\s*(Approve|Reject)', ceo_vote_response, re.IGNORECASE)
        ceo_reason_match = re.search(r'REASON:\s*(.*?)(?:$|\n|\r)', ceo_vote_response, re.DOTALL)

        if ceo_vote_match and ceo_reason_match:
            ceo_vote = ceo_vote_match.group(1)
            ceo_vote_reason = ceo_reason_match.group(1).strip()
        else:
            # Fall back to model-based extraction
            extracted_fields = extract_fields_with_model(
                ceo_vote_response,
                format_example,
                fields_to_extract,
                model=model_low
            )

            ceo_vote = extracted_fields.get("VOTE", "Approve")  # Default to Approve for CEO if parsing fails
            ceo_vote_reason = extracted_fields.get("REASON", "No reason provided")

            # Validate vote is either Approve or Reject
            if ceo_vote and not any(v.lower() == ceo_vote.lower() for v in ["Approve", "Reject"]):
                # If vote is invalid, use sample_choice for additional validation
                try:
                    vote_validation_prompt = f"""Based only on this response, is the CEO voting to approve or reject the proposal?

Response: {ceo_vote_response}

Choose one: Approve or Reject
"""
                    idx, ceo_vote, _ = model_low.sample_choice(
                        vote_validation_prompt,
                        ["Approve", "Reject"]
                    )
                except Exception as e:
                    print(f"Error validating CEO vote: {e}")
                    ceo_vote = "Approve"  # Default if all extraction methods fail

        votes[ceo_name] = ceo_vote
        vote_reasons[ceo_name] = ceo_vote_reason
        # Record CEO vote in decision tracker
        if hasattr(self.company, 'decision_tracker'):
            self.company.decision_tracker.record_agent_vote(
                meeting_id,
                ceo_name,
                ceo_vote,
                ceo_vote_reason,
                is_first_vote=(not is_alternative_proposal)
            )
        # Announce CEO vote with reason
        ceo_vote_announcement = f"{ceo_name} (CEO) votes: {ceo_vote}. Reason: {ceo_vote_reason}"
        for sh in self.company.shareholders.values():
            sh.agent.observe(ceo_vote_announcement)
        self.company.ceo.agent.observe(ceo_vote_announcement)

        # Use VotingManager to record and process votes
        stats = self.voting_manager.record_votes(
            meeting_id,
            proposal,
            proposal_speech,
            votes,
            vote_reasons,
            proposal_type
        )

        # Get vote summary using VotingManager for documenting full results
        votes_text = self.voting_manager.get_summary_text(meeting_id, proposal_type)

        # Add vote results to all meeting documents (visible to everyone)
        if hasattr(self.document_manager, 'add_to_all_documents'):
            self.document_manager.add_to_all_documents(votes_text, tags=[
                "voting_results" if proposal_type == "initial" else "alternative_voting_results"])

        # Determine if proposal passed using VotingManager
        passed = self.voting_manager.get_vote_result(meeting_id, proposal_type)

        # Extract year and quarter from meeting_id for tagging (for memory storage)
        year = int(meeting_id.split('_')[2].replace('year', ''))
        quarter = None
        if 'q' in meeting_id:
            quarter = int(meeting_id.split('_')[-1].replace('q', ''))

        # Add personalized vote summary to each agent's memory
        for agent_name in list(self.company.shareholders.keys()) + [ceo_name]:
            # Get personalized vote summary for this agent
            personal_vote_summary = self.voting_manager.get_agent_vote_summary(meeting_id, agent_name, proposal_type)

            # Add to agent's memory with appropriate tags
            self.document_manager.add_to_agent_memory(
                agent_name,
                personal_vote_summary,
                tags=[meeting_type, f"year_{year}",
                      f"quarter_{quarter}" if quarter else "annual",
                      "my_vote", meeting_id, f"{proposal_type}_vote"]
            )

        # Maintain backward compatibility with discussion_log
        if proposal_type == "initial":
            discussion_log["votes"] = votes
            discussion_log["vote_reasons"] = vote_reasons
            discussion_log["passed"] = passed
        else:
            discussion_log["alternative_votes"] = votes
            discussion_log["alternative_vote_reasons"] = vote_reasons
            discussion_log["alternative_passed"] = passed

        # Verify votes were properly recorded (for debugging)
        if not self.voting_manager.has_votes(meeting_id, proposal_type):
            print(f"WARNING: Votes for {meeting_id} ({proposal_type}) were not properly recorded!")

        return votes, vote_reasons, passed

    def _get_ceo_decision_after_failed_vote(self, meeting_id, meeting_type, rejected_proposal):
        """Helps CEO decide how to proceed after a proposal fails to pass."""
        # Get the CEO's perspective context
        ceo_context = ""
        if hasattr(self.document_manager, 'get_agent_context'):
            ceo_context = self.document_manager.get_agent_context(self.company.ceo.config.name)

        # Get CEO's position summary using the new method
        ceo_name = self.company.ceo.config.name
        position_summary = self.position_manager.get_position_summary_text(meeting_id, ceo_name)

        # Get decision context from voting manager
        decision_context = self.voting_manager.get_ceo_decision_context(meeting_id)

        spec = entity.free_action_spec(
            call_to_action=f"""You are {self.company.ceo.config.name}, the CEO of the investment company with MBTI personality type {self.company.ceo.config.traits}. 

{self.CEO_DISCUSSION_GOAL}

Your proposal '{rejected_proposal}' was rejected in the vote.

{decision_context}

{position_summary}

You now face a leadership challenge that will reveal your character and leadership style. You have two options:

1. Force Through the Original Proposal: As CEO, you can decide to implement your original proposal despite objections. This would demonstrate conviction in your vision and might be appropriate if you strongly believe it's the best decision for the company, but it could damage shareholder relations.

2. Create a Modified Proposal: You can develop a new proposal that addresses shareholder concerns. This demonstrates responsiveness and maintains a collaborative approach, but may lead to a compromise that isn't optimal in your view.

Based on your MBTI personality type ({self.company.ceo.config.traits}), consider:
- How do your personality traits typically respond to disagreement?
- How important is maintaining your original position vs. finding consensus?
- Which approach feels most authentic to your leadership style?
- Which approach would best serve the company in this situation?

IMPORTANT: Address all meeting participants in first person ("I/my"). Never refer to yourself by name or in third person. Maintain a professional tone suitable for addressing the entire group.

Your response MUST follow this exact format:
DECISION: [Force through original proposal/Create modified proposal]
REASON: [Your explanation that reflects your MBTI personality and leadership style]

IMPORTANT FORMAT INSTRUCTION: Your response MUST strictly follow the exact output format specified above. Do not add any explanations, notes, or content outside this format. Any deviation from the required format structure will cause processing errors. Provide ONLY the formatted response as outlined.

Your explanation should consider:
- How your personality traits influence your response to this situation
- The specific shareholder concerns that factored into your decision
- The balance between asserting your leadership vision and respecting shareholder input
- Your assessment of what will ultimately lead to the best outcome for the company

This explanation will be shared with the shareholders, so frame it in a way that reflects your genuine leadership approach.

===MEETING CONTEXT===
    {ceo_context}
=====================""",
            tag=f"{meeting_type}_after_failed_vote_decision_with_reason"
        )

        response = self.company.ceo.agent.act(spec)

        # Extract decision and reason using extract_fields_with_model
        format_example = """DECISION: [Force through original proposal/Create modified proposal]
REASON: [Your explanation that reflects your MBTI personality and leadership style]"""

        fields_to_extract = ["DECISION", "REASON"]
        extracted_fields = extract_fields_with_model(
            response,
            format_example,
            fields_to_extract,
            model=model_low
        )

        # Get extracted fields with defaults if extraction failed
        decision = extracted_fields.get("DECISION", "Create modified proposal")
        reason = extracted_fields.get("REASON", "The shareholders have raised valid concerns that should be addressed.")
        # Record post-vote decision
        if hasattr(self.company, 'decision_tracker'):
            self.company.decision_tracker.record_post_vote_decision(meeting_id, decision, reason)
        # Simple validation - no model call needed
        valid_decisions = ["Force through original proposal", "Create modified proposal"]

        # Fix for the NoneType error - check if decision is a string before calling lower()
        if isinstance(decision, str):
            # Check if decision matches any valid option
            if not any(valid.lower() in decision.lower() for valid in valid_decisions):
                # Default to more consensus-oriented option if validation fails
                decision = "Create modified proposal"
        else:
            # If decision is None or not a string, use default
            decision = "Create modified proposal"

        return decision, reason

    def _get_ceo_alternative_proposal(self, meeting_id, meeting_type, options, discussion_log):
        """Helps CEO formulate an alternative proposal after a failed vote."""

        # Get the CEO's perspective context
        ceo_context = ""
        if hasattr(self.document_manager, 'get_agent_context'):
            ceo_context = self.document_manager.get_agent_context(self.company.ceo.config.name)

        # Get CEO's name
        ceo_name = self.company.ceo.config.name

        # Get CEO's position summary using the new method
        position_summary = self.position_manager.get_position_summary_text(meeting_id, ceo_name)

        # Get CEO's initial preference if available using position manager
        initial_position_data = self.position_manager.get_initial_position(meeting_id, ceo_name)
        ceo_preference = initial_position_data['option'] if initial_position_data else ""

        original_proposal = discussion_log["proposal"]["option"]

        # Get comprehensive voting history for context - simplify to just this one
        voting_history = self.voting_manager.get_comprehensive_voting_history(meeting_id)

        # Get remaining options list for clear reminder
        remaining_options = [opt for opt in options.keys() if opt != original_proposal]
        valid_options_list = ", ".join([f"'{opt}'" for opt in remaining_options])
        options_reminder = f"CRITICAL: Your alternative proposal MUST be one of these valid options: {valid_options_list}. You cannot propose a hybrid or compromise option outside our predefined choices."

        spec = entity.free_action_spec(
            call_to_action=f"""You are {self.company.ceo.config.name}, the CEO of the investment company with MBTI personality type {self.company.ceo.config.traits}. 

{self.CEO_DISCUSSION_GOAL}

Your initial proposal '{original_proposal}' was rejected in the vote.

{voting_history}

You've chosen to create a modified proposal rather than force through your original choice. This demonstrates responsiveness while maintaining your leadership role.

LEADERSHIP OPPORTUNITY:
This is a critical opportunity to show adaptive leadership while staying true to your core vision and MBTI personality type. You need to select an alternative proposal that addresses key shareholder concerns while still aligning with your assessment of what's best for the company.

Available options (excluding your original proposal):
{valid_options_list}

{options_reminder}

{position_summary}

DECISION CONTEXT:
Your initial preference before any discussion began was {ceo_preference}, and your first proposal was {original_proposal}. Consider how your alternative proposal should relate to these earlier positions given your personality type and leadership style.
IMPORTANT: Deliver your formal speech in first person ("I/my" and when appropriate "we/our"). Never refer to yourself by name or in third person.
Consider:
1. Which alternative option best addresses the specific concerns raised during voting?
2. How can you maintain your leadership vision while showing appropriate flexibility?
3. Which option would be most defensible in a second vote?
4. What option aligns best with your MBTI personality type's approach to compromise?
5. How does your initial preference ({ceo_preference}) influence your current thinking?

Remember:
- You must select a different option than your original proposal
- Your alternative MUST be one of our valid predefined options - do not create hybrid solutions or new approaches
- This proposal will go to another shareholder vote requiring 2/3 majority
- If this proposal also fails, you'll need to make a final executive decision

Your response MUST follow this exact format:
ALTERNATIVE PROPOSAL: [Option name exactly as it appears in the options list]
PROPOSAL REASONING: [Your explanation of why this alternative addresses shareholder concerns while serving company needs]

Example format:
ALTERNATIVE PROPOSAL: Option B
PROPOSAL REASONING: After considering the valid concerns about risk exposure raised during our vote, I believe Option B offers the best balance of addressing these concerns while still positioning us for appropriate growth in the current market conditions.

IMPORTANT FORMAT INSTRUCTION: Your response MUST strictly follow the exact output format specified above. Do not add any explanations, notes, or content outside this format. Any deviation from the required format structure will cause processing errors. Provide ONLY the formatted response as outlined.

===MEETING CONTEXT===
    {ceo_context}
=====================""",
            tag=f"{meeting_type}_alternative_proposal_with_reason"
        )

        response = self.company.ceo.agent.act(spec)

        # Extract alternative proposal and reasoning using extract_fields_with_model
        format_example = """ALTERNATIVE PROPOSAL: [Option name exactly as it appears in the options list]
PROPOSAL REASONING: [Your explanation of why this alternative addresses shareholder concerns while serving company needs]"""

        fields_to_extract = ["ALTERNATIVE_PROPOSAL", "PROPOSAL_REASONING"]
        extracted_fields = extract_fields_with_model(
            response,
            format_example,
            fields_to_extract,
            model=model_low
        )

        # Get extracted fields with defaults
        alternative = extracted_fields.get("ALTERNATIVE_PROPOSAL")
        reason = extracted_fields.get("PROPOSAL_REASONING",
                                      "Based on shareholder feedback, this alternative better addresses our collective concerns.")

        # Simple validation - no model call needed
        if not alternative or alternative not in options or alternative == original_proposal:
            # Default to first available alternative
            alternative = next((opt for opt in options.keys() if opt != original_proposal), list(options.keys())[0])

        # Notify all participants of the new proposal
        proposal_announcement = f"CEO Alternative Proposal: Since the previous proposal was not approved, I now propose {alternative}. Reason: {reason}"
        # Record second proposal
        if hasattr(self.company, 'decision_tracker'):
            self.company.decision_tracker.record_second_proposal(meeting_id, alternative, reason)
        for shareholder in self.company.shareholders.values():
            shareholder.agent.observe(proposal_announcement)

        self.company.ceo.agent.observe(proposal_announcement)

        return alternative, reason

    def _get_ceo_closing_statement_after_vote(self, meeting_id, option):
        """Helps CEO craft a closing statement after voting on a proposal."""
        # Get the CEO's perspective context
        ceo_context = ""
        if hasattr(self.document_manager, 'get_agent_context'):
            ceo_context = self.document_manager.get_agent_context(self.company.ceo.config.name)

        # Get CEO's name
        ceo_name = self.company.ceo.config.name

        # Get CEO's position summary using the new method
        position_summary = self.position_manager.get_position_summary_text(meeting_id, ceo_name)

        # Determine if this is an alternative proposal
        is_alternative = (meeting_id in self.voting_manager.votes and
                          "alternative" in self.voting_manager.votes[meeting_id] and
                          self.voting_manager.votes[meeting_id]["alternative"]["proposal"] == option)

        proposal_type = "alternative" if is_alternative else "initial"

        # Get vote statistics from voting manager
        vote_stats = self.voting_manager.get_vote_counts(meeting_id, proposal_type)

        # Get appropriate vote summary based on proposal type
        if is_alternative:
            # For alternative proposals, get comprehensive voting history
            vote_summary = self.voting_manager.get_comprehensive_voting_history(meeting_id)

            # Add context about this being an alternative proposal
            proposal_context = """
Note: This is the ALTERNATIVE proposal that passed after the initial proposal was rejected.
The shareholders have now approved your second proposal after rejecting your first choice.
This demonstrates your ability to adapt and find compromise solutions.
"""
        else:
            # For initial proposals, just get the standard summary
            vote_summary = self.voting_manager.get_summary_text(meeting_id, proposal_type, include_reasons=True)
            proposal_context = ""

        spec = entity.free_action_spec(
            call_to_action=f"""You are {self.company.ceo.config.name}, the CEO of the investment company with MBTI personality type {self.company.ceo.config.traits}. 

{self.CEO_DISCUSSION_GOAL}

The proposal '{option}' has PASSED with {vote_stats['approve_count']} votes in favor and {vote_stats['reject_count']} votes against ({vote_stats['approve_percentage']:.1f}% approval).

{proposal_context}

{vote_summary}

{position_summary}

LEADERSHIP OPPORTUNITY:
This is your chance to demonstrate authentic leadership by delivering a closing statement that:
1. Acknowledges the outcome of the vote in a way authentic to your MBTI personality type
2. Addresses both strengths and areas for improvement
3. Outlines 2-3 specific commitments for improvement in the coming year
4. Ends with a forward-looking vision that inspires confidence

IMPORTANT: Deliver your formal speech in first person ("I/my" and when appropriate "we/our"). Never refer to yourself by name or in third person.
Keep your closing statement concise (4-6 sentences) and focused on what these ratings mean for your leadership and the company's future.

===MEETING CONTEXT===
    {ceo_context}
=====================""",
            tag=f"{meeting_id}_closing_statement"
        )

        closing = self.company.ceo.agent.act(spec)
        if hasattr(self.company, 'decision_tracker'):
            self.company.decision_tracker.record_closing_statement(meeting_id, closing)
        # Notify all participants
        for shareholder in self.company.shareholders.values():
            shareholder.agent.observe(closing)

        self.company.ceo.agent.observe(closing)

        return closing

    def _get_ceo_closing_statement_after_force_through(self, meeting_id, option):
        """Helps CEO craft a closing statement after forcing through a rejected proposal."""

        # Get the CEO's perspective context
        ceo_context = ""
        if hasattr(self.document_manager, 'get_agent_context'):
            ceo_context = self.document_manager.get_agent_context(self.company.ceo.config.name)

        # Get CEO's name
        ceo_name = self.company.ceo.config.name

        # Get CEO's position summary using the new method
        position_summary = self.position_manager.get_position_summary_text(meeting_id, ceo_name)

        # Get vote statistics from voting manager
        vote_stats = self.voting_manager.get_vote_counts(meeting_id, "initial")
        rejection_percentage = vote_stats["reject_percentage"]

        spec = entity.free_action_spec(
            call_to_action=f"""You are {self.company.ceo.config.name}, the CEO of the investment company with MBTI personality type {self.company.ceo.config.traits}. 

{self.CEO_DISCUSSION_GOAL}

You have decided to force through your proposal '{option}' despite it being rejected in the vote with {vote_stats['approve_count']} votes in favor and {vote_stats['reject_count']} votes against ({rejection_percentage:.1f}% rejection).

{position_summary}

DEFINING LEADERSHIP MOMENT:
This is perhaps the strongest assertion of your leadership in the entire process. You're overriding majority opinion because you believe this decision is genuinely best for the company. This requires:

1. Acknowledging the opposition with respect (not dismissal)
2. Articulating why you believe this decision is necessary despite objections
3. Addressing the specific concerns raised by shareholders
4. Outlining how you plan to mitigate the risks they identified
5. Ending on a forward-looking note that maintains team cohesion

IMPORTANT: Deliver your formal speech in first person ("I/my" and when appropriate "we/our"). Never refer to yourself by name or in third person.
Your statement must balance conviction with empathy, authority with respect. The way you handle this moment will have lasting impact on your perceived leadership capabilities and shareholder relationships.

Craft a 3-5 sentence closing statement that would be appropriate for a formal boardroom setting, ending with a clear indication that this is your final decision as CEO.

===MEETING CONTEXT===
    {ceo_context}
=====================""",
            tag=f"{meeting_id}_force_through_closing"
        )

        closing = self.company.ceo.agent.act(spec)
        if hasattr(self.company, 'decision_tracker'):
            self.company.decision_tracker.record_closing_statement(meeting_id, closing)
        # Notify all participants
        for shareholder in self.company.shareholders.values():
            shareholder.agent.observe(closing)

        self.company.ceo.agent.observe(closing)

        return closing

    def _get_ceo_executive_decision(self, meeting_id, meeting_type, options):
        """Helps CEO make a final executive decision when all proposals have failed."""

        # Get the CEO's perspective context
        ceo_context = ""
        if hasattr(self.document_manager, 'get_agent_context'):
            ceo_context = self.document_manager.get_agent_context(self.company.ceo.config.name)

        # Get CEO's full position summary
        ceo_name = self.company.ceo.config.name
        position_summary = self.position_manager.get_position_summary_text(meeting_id, ceo_name)

        # Get voting summary directly from voting manager
        voting_summary = self.voting_manager.get_comprehensive_voting_history(meeting_id)

        spec = entity.free_action_spec(
            call_to_action=f"""You are {ceo_name}, the CEO of the investment company with MBTI personality type {self.company.ceo.config.traits}. 

{self.CEO_DISCUSSION_GOAL}

EXTRAORDINARY LEADERSHIP MOMENT:
Both of your proposals have been rejected by the shareholders:
{voting_summary}

{position_summary}

This is the defining moment of your leadership. As CEO, you must now make an executive decision on your own authority. The entire decision process has run its course:
- You've led thorough discussions exploring all perspectives
- You've proposed two different solutions
- Both have failed to achieve sufficient consensus
- You must now decide independently what is best for the company

REFLECTION POINTS:
1. What has the discussion process revealed about the true needs of the company?
2. Which option, based on all you've heard, genuinely serves the company best?
3. How does your MBTI personality type ({self.company.ceo.config.traits}) influence your final decision-making approach in this challenging situation?
4. Should you return to your initial assessment, choose one of your rejected proposals despite objections, or select an entirely different option?
5. What will demonstrate the most authentic and effective leadership in this moment?

IMPORTANT: Address all meeting participants in first person ("I/my"). Never refer to yourself by name or in third person. Maintain a professional tone suitable for addressing the entire group.
You must select one option from: {', '.join(options.keys())}

Your response MUST follow this exact format:
EXECUTIVE DECISION: [Option name exactly as it appears in the options list]
DECISION REASONING: [Your explanation of why this is your final decision, reflecting your authentic leadership style]

Example format:
EXECUTIVE DECISION: Option C
DECISION REASONING: After considering all perspectives across our multiple discussions and votes, I'm convinced Option C represents the most balanced approach for our company in the current market environment. While I acknowledge the concerns about risk exposure, my responsibility as CEO requires prioritizing our long-term strategic positioning over short-term comfort.

IMPORTANT FORMAT INSTRUCTION: Your response MUST strictly follow the exact output format specified above. Do not add any explanations, notes, or content outside this format. Any deviation from the required format structure will cause processing errors. Provide ONLY the formatted response as outlined.

===MEETING CONTEXT===
    {ceo_context}
=====================""",
            tag=f"{meeting_type}_executive_decision_with_reason"
        )

        response = self.company.ceo.agent.act(spec)

        # Define format example and fields to extract
        format_example = """EXECUTIVE DECISION: [Option name exactly as it appears in the options list]
DECISION REASONING: [Your explanation of why this is your final decision, reflecting your authentic leadership style]"""

        fields_to_extract = ["EXECUTIVE_DECISION", "DECISION_REASONING"]

        # Use extract_fields_with_model instead of regex
        extracted_fields = extract_fields_with_model(
            response,
            format_example,
            fields_to_extract,
            model=model_low
        )

        # Get decision and reasoning with fallbacks
        decision = extracted_fields.get("EXECUTIVE_DECISION")
        reason = extracted_fields.get("DECISION_REASONING",
                                      "After careful consideration of all perspectives, this option best serves our company's needs.")

        # Validate decision is one of the valid options
        if not decision or decision not in options:
            # Default to first option if invalid or missing
            decision = list(options.keys())[0]
            print(f"Warning: CEO executive decision '{decision}' not found in valid options. Defaulting to {decision}.")

        # Notify all participants of the executive decision
        announcement = f"CEO Executive Decision: As we could not reach consensus after two votes, as CEO I have decided to choose {decision}. {reason}"
        if hasattr(self.company, 'decision_tracker'):
            self.company.decision_tracker.record_executive_decision(meeting_id, decision)
        for shareholder in self.company.shareholders.values():
            shareholder.agent.observe(announcement)

        self.company.ceo.agent.observe(announcement)

        return decision, reason

    def _define_evaluation_criteria(self):

        evaluation_criteria = {
            "ceo": {
                "leadership_effectiveness": {
                    "title": "Leadership Effectiveness",
                    "description": "Ability to guide discussions, balance shareholder input, and maintain clear direction",
                    "question": "How effective was the CEO in leading the group toward optimal decisions?",
                    "scale": {
                        1: "Ineffective leadership lacking direction",
                        2: "Below average leadership with inconsistent direction",
                        3: "Average leadership maintaining basic direction",
                        4: "Strong leadership with clear direction",
                        5: "Exceptional leadership that maximized group effectiveness"
                    }
                },
                "decision_quality": {
                    "title": "Decision Quality",
                    "description": "Soundness of final decisions considering available information and market conditions",
                    "question": "How well did the CEO's decisions serve the company's interests?",
                    "scale": {
                        1: "Poor decisions that ignored key information",
                        2: "Below average decisions with limited consideration of information",
                        3: "Average decisions with adequate consideration of information",
                        4: "Good decisions that effectively balanced most factors",
                        5: "Excellent decisions that optimally balanced risk and return"
                    }
                },
                "communication_clarity": {
                    "title": "Communication Clarity",
                    "description": "Clarity in explaining investment options, sharing performance data, and articulating reasoning",
                    "question": "How clearly did the CEO explain complex information and decision rationales?",
                    "scale": {
                        1: "Consistently unclear or confusing communication",
                        2: "Often unclear communication requiring clarification",
                        3: "Adequately clear communication",
                        4: "Very clear communication with minor issues",
                        5: "Exceptionally clear and effective communication"
                    }
                },
                "responsiveness_to_input": {
                    "title": "Responsiveness to Input",
                    "description": "Willingness to consider shareholder perspectives and incorporate valuable insights",
                    "question": "How well did the CEO respond to and incorporate your input?",
                    "scale": {
                        1: "Dismissed or ignored shareholder input",
                        2: "Minimally acknowledged input without incorporation",
                        3: "Considered input but limited incorporation",
                        4: "Actively incorporated valuable input",
                        5: "Thoughtfully considered and integrated diverse perspectives"
                    }
                },
                "adaptability": {
                    "title": "Adaptability",
                    "description": "Ability to adjust approach based on changing market conditions or new information",
                    "question": "How effectively did the CEO adapt strategies to changing circumstances?",
                    "scale": {
                        1: "Rigid adherence to initial positions regardless of new information",
                        2: "Limited adaptation to changing conditions",
                        3: "Adequate adaptation to major changes",
                        4: "Good adaptation while maintaining consistency",
                        5: "Highly adaptable while maintaining consistent strategic vision"
                    }
                }
            },
            "company": {
                "financial_results": {
                    "title": "Financial Results",
                    "description": "Actual investment returns compared to market opportunities",
                    "question": "How well did the company perform financially relative to market conditions?",
                    "scale": {
                        1: "Significantly underperformed market opportunities",
                        2: "Somewhat underperformed market opportunities",
                        3: "Achieved average returns given market conditions",
                        4: "Outperformed market average",
                        5: "Captured optimal or near-optimal returns given market conditions"
                    }
                },
                "risk_management": {
                    "title": "Risk Management",
                    "description": "Appropriate balance of risk and return across investment decisions",
                    "question": "How effectively did the company balance risk and potential returns?",
                    "scale": {
                        1: "Consistently poor risk-reward decisions",
                        2: "Often imbalanced risk-reward decisions",
                        3: "Adequate risk-reward balance overall",
                        4: "Good risk-reward balance with few exceptions",
                        5: "Excellent risk-reward balance across all quarters"
                    }
                },
                "strategic_alignment": {
                    "title": "Strategic Alignment",
                    "description": "Consistency of investment decisions with stated strategies and market conditions",
                    "question": "How well did investment choices align with the company's stated approach?",
                    "scale": {
                        1: "Investments frequently contradicted stated strategy",
                        2: "Limited alignment between investments and strategy",
                        3: "Moderate alignment with occasional deviations",
                        4: "Strong alignment with minor inconsistencies",
                        5: "Perfect alignment between strategy and execution"
                    }
                },
                "capital_utilization": {
                    "title": "Capital Utilization",
                    "description": "Efficiency in using available capital to generate returns",
                    "question": "How efficiently did the company utilize its available capital?",
                    "scale": {
                        1: "Poor utilization leaving significant opportunity untapped",
                        2: "Below average capital utilization",
                        3: "Average efficiency in capital deployment",
                        4: "Good optimization of capital resources",
                        5: "Optimal utilization of capital resources"
                    }
                },
                "market_responsiveness": {
                    "title": "Market Responsiveness",
                    "description": "Ability to adjust investment strategy based on changing market conditions",
                    "question": "How effectively did the company adapt to the specific market environment?",
                    "scale": {
                        1: "Failed to adapt to clear market signals",
                        2: "Slow or limited adaptation to market conditions",
                        3: "Adequate responsiveness to major market shifts",
                        4: "Good responsiveness to market conditions",
                        5: "Expertly navigated changing market conditions"
                    }
                }
            }
        }

        return evaluation_criteria

    def _create_meeting_summaries_for_review(self, year):

        # Find all meeting IDs for the specified year
        budget_meeting_id = f"annual_budget_year{year}"
        quarterly_meeting_ids = [f"quarterly_investment_year{year}_q{q}" for q in range(1, 5)]
        all_meeting_ids = [budget_meeting_id] + quarterly_meeting_ids

        # Initialize summary structure
        summary = {
            "overall_summary": "",
            "budget_option": "Unknown",
            "quarterly_decisions": [],
            "quarterly_assets": getattr(self.company, "assets_logs", [])[year * 4:(year + 1) * 4] if hasattr(
                self.company, "assets_logs") else []
        }

        # Get budget meeting data using PositionManager (if available)
        if hasattr(self, 'position_manager') and self.position_manager.positions.get(budget_meeting_id):
            # Get CEO's final position as the budget decision
            ceo_name = self.company.ceo.config.name if hasattr(self.company, 'ceo') else "CEO"
            ceo_position = self.position_manager.get_current_position(budget_meeting_id, ceo_name)
            if ceo_position:
                summary["budget_option"] = ceo_position["option"]
        else:
            # Legacy fallback - find in meeting logs
            relevant_logs = [log for log in self.company.meeting_logs
                             if isinstance(log, dict) and "meeting_id" in log
                             and budget_meeting_id in log.get("meeting_id", "")]
            budget_log = next(iter(relevant_logs), None)
            if budget_log:
                summary["budget_option"] = budget_log.get("budget_option", "Unknown")

        # Get quarterly investment decisions
        for quarter in range(1, 5):
            quarterly_meeting_id = f"quarterly_investment_year{year}_q{quarter}"

            # Initialize decision data
            decision = {
                "quarter": quarter,
                "asset": "Unknown",
                "discussion_rounds": 0,
                "passed_first_vote": False
            }

            # Try to get data from PositionManager
            if hasattr(self, 'position_manager') and self.position_manager.positions.get(quarterly_meeting_id):
                ceo_name = self.company.ceo.config.name if hasattr(self.company, 'ceo') else "CEO"
                ceo_position = self.position_manager.get_current_position(quarterly_meeting_id, ceo_name)
                if ceo_position:
                    decision["asset"] = ceo_position["option"]

                # Count discussion rounds from position_manager
                ceo_history = self.position_manager.get_position_history(quarterly_meeting_id, ceo_name)
                decision["discussion_rounds"] = max([p.get("round", 0) for p in ceo_history]) if ceo_history else 0

            # Try to get voting data from VotingManager
            if hasattr(self, 'voting_manager') and self.voting_manager.has_votes(quarterly_meeting_id, "initial"):
                decision["passed_first_vote"] = self.voting_manager.get_vote_result(quarterly_meeting_id, "initial")

            # If we couldn't get data from managers, try legacy approach
            if decision["asset"] == "Unknown":
                # Legacy fallback - find in meeting logs
                relevant_logs = [log for log in self.company.meeting_logs
                                 if isinstance(log, dict) and "meeting_id" in log
                                 and quarterly_meeting_id in log.get("meeting_id", "")]
                q_log = next(iter(relevant_logs), None)
                if q_log:
                    decision["asset"] = q_log.get("proposal", {}).get("option", "Unknown")
                    decision["discussion_rounds"] = len(q_log.get("rounds", []))
                    decision["passed_first_vote"] = q_log.get("passed", False)

            # Add to quarterly decisions
            summary["quarterly_decisions"].append(decision)

        # Sort by quarter
        summary["quarterly_decisions"].sort(key=lambda x: x["quarter"])

        # Create quarterly summary text
        quarterly_summary = "Quarterly investment decisions:\n"
        for decision in summary["quarterly_decisions"]:
            quarterly_summary += f"- Q{decision['quarter']}: {decision['asset']}"
            if decision["asset"] != "Unknown":
                quarterly_summary += f" (after {decision['discussion_rounds']} discussion rounds)"
                quarterly_summary += f", {'approved' if decision['passed_first_vote'] else 'initially rejected'}"
            quarterly_summary += "\n"

        # Generate overall summary (use model only for high-level interpretation if needed)
        summary_prompt = f"""Create a brief summary of the company's Year {year} performance:
            1. What were the key investment decisions made?
            2. How would you characterize the decision-making process?

            {quarterly_summary}
            Keep the summary very brief (100 words maximum).
        """

        try:
            overall_summary = self.model.sample_text(
                summary_prompt,
                max_tokens=150,
                temperature=0.3
            )
            summary["overall_summary"] = overall_summary
        except:
            # Fallback: create a simple summary without model call
            summary[
                "overall_summary"] = f"Year {year} included a budget decision ({summary['budget_option']}) and quarterly investments in various assets including: " + ", ".join(
                [d["asset"] for d in summary["quarterly_decisions"] if d["asset"] != "Unknown"])

        return summary

    # Methods for annual review meeting
    def _get_ceo_opening_statement_for_review(self, meeting_id, year_performance, evaluation_criteria=None):
        """Helps CEO craft an opening statement for the annual review meeting."""

        # Get the CEO's perspective context
        ceo_context = ""
        if hasattr(self.document_manager, 'get_agent_context'):
            ceo_context = self.document_manager.get_agent_context(self.company.ceo.config.name)

        # Extract year from meeting_id
        year = int(meeting_id.split('_')[2].replace('year', ''))

        # ==== SECTION FROM ORIGINAL _get_ceo_opening_statement FUNCTION ====
        # Annual Review Meeting
        welcome = f"Welcome to our annual review meeting for Year {year}. Thank you all for your participation and commitment throughout this year."

        # Get year performance data
        starting_assets = year_performance.get('starting_assets', 0)
        ending_assets = year_performance.get('ending_assets', self.company.assets)

        # First describe market environment
        market_condition = year_performance.get('market_condition', '')
        market_intro = self.market_condition_intros.get(market_condition, "")
        market_context = f"During year {year}, we operated in a {market_condition} market environment. {market_intro}"

        overall_performance = f"We began the year with ${starting_assets:.2f} in assets and ended with ${ending_assets:.2f}, representing a {((ending_assets / starting_assets) - 1) * 100:.2f}% change in our company's value."

        # Get current year's market performance data
        end_quarter = year * 4

        # Extract only the current year's data from the full history
        full_history = self.company.market.get_formatted_historical_returns(end_quarter)

        # Find where the current year starts in the full history
        current_year_marker = f"Year {year} ({1970 + year})"
        current_year_start = full_history.find(current_year_marker)

        if current_year_start != -1:
            current_year_data = "\n\n" + full_history[current_year_start:]
        else:
            current_year_data = "\n\nNo market performance data available for the current year."

        # Add comparison between our decisions and what we could have earned
        our_decisions = [inv for inv in self.company.investment_decisions if inv["year"] == year]

        comparison_data = "\n\nComparison between our investment choices and potential alternatives:\n"
        total_actual_return = 0
        best_possible_return = 0

        for q in range(1, 5):
            quarter_idx = (year - 1) * 4 + q - 1
            quarter_returns = self.company.market.get_returns_for_quarter(quarter_idx)

            # Find our decision for this quarter
            our_decision = next((inv for inv in our_decisions if inv["quarter"] == q), None)
            if our_decision:
                our_asset = our_decision["asset"]
                our_budget = our_decision["budget"]
                our_return = our_decision["return_amount"]
                total_actual_return += our_return

                # Find best possible asset for this quarter
                best_asset = max(quarter_returns.items(), key=lambda x: x[1])
                best_possible_return_amount = our_budget * best_asset[1]
                best_possible_return += best_possible_return_amount

                comparison_data += f"Quarter {q}: We invested ${our_budget:.2f} in {our_asset} for a return of ${our_return:.2f} "
                comparison_data += f"({quarter_returns[our_asset] * 100:.1f}%)\n"
                comparison_data += f"  Best option was {best_asset[0]} with {best_asset[1] * 100:.1f}% return "
                comparison_data += f"(would have yielded ${best_possible_return_amount:.2f})\n"

        if our_decisions:
            efficiency = (total_actual_return / best_possible_return) * 100 if best_possible_return > 0 else 0
            comparison_data += f"\nOverall, our decisions captured {efficiency:.1f}% of the maximum possible returns this year.\n"

        # Add comprehensive performance history
        performance_history = self.company.get_formatted_annual_performance_history(year)

        # Use pre-calculated performance analysis to avoid LLM call
        performance_analysis = f"\n\nMy assessment of our performance: This year presented various challenges and opportunities under the {market_condition} market conditions. Our investment strategy aimed to balance risk and return, resulting in a {((ending_assets / starting_assets) - 1) * 100:.2f}% change in company value. While there were quarters where we could have achieved higher returns with different asset allocations, our decisions were made with the information available at each decision point."

        quarterly_summary = f"\n\nLet me summarize our quarterly investments for the year:\n"
        for q_data in year_performance.get('quarterly_investments', []):
            quarterly_summary += f"- Q{q_data['quarter']}: Invested ${q_data['budget']:.2f} in {q_data['asset']}, yielding a {q_data['return_rate'] * 100:.1f}% return (${q_data['return_amount']:.2f}).\n"

        # ==== APPEND EVALUATION CRITERIA FROM _get_ceo_opening_statement_for_review ====
        # Create evaluation dimensions text if criteria provided
        evaluation_dimensions = ""
        if evaluation_criteria:
            evaluation_dimensions += "\n\nFor my leadership assessment, I'd like your feedback across five key dimensions:\n"
            for key, criteria in evaluation_criteria["ceo"].items():
                evaluation_dimensions += f"- {criteria['title']}: {criteria['description']}\n"

            evaluation_dimensions += "\nFor company performance assessment, please consider five dimensions:\n"
            for key, criteria in evaluation_criteria["company"].items():
                evaluation_dimensions += f"- {criteria['title']}: {criteria['description']}\n"

        # ==== ADD MEETING PROCESS INFORMATION ====
        meeting_process = """
The process for today's annual review meeting will be as follows:

1. Written Evaluation Submission: Each shareholder will submit a written evaluation of both the company's performance and my leadership effectiveness.

2. CEO Response: After reviewing all submissions, I will address the key themes and respond to specific questions raised.

3. Performance Ratings: Each shareholder will then provide numerical ratings (on a 1-10 scale) for both company performance and CEO leadership across several dimensions.

4. Meeting Conclusion: I will summarize the feedback and ratings, and outline improvement plans for the coming year.
    """

        meeting_purpose = """
The purpose of today's meeting is to:

1. Collectively evaluate our company's overall performance this year
2. Assess the effectiveness of our investment strategy
3. Evaluate my leadership as your CEO
4. Identify lessons learned to apply in the coming year

I invite each of you to submit your written evaluations covering both the company's performance and your assessment of my leadership, including numerical ratings on a scale of 1-10. These evaluations will help guide our strategy moving forward.

After you've submitted your assessments, I'll address the key themes and outline my thoughts on how we can improve in the coming year."""

        # Combine all sections from both functions
        opening_statement = welcome + "\n\n" + market_context + "\n\n" + overall_performance + current_year_data + comparison_data + "\n\n" + performance_history + "\n\n" + performance_analysis + quarterly_summary + "\n\n" + evaluation_dimensions + "\n" + meeting_process + "\n" + meeting_purpose

        # Add the statement to CEO's memory document
        if hasattr(self.document_manager, 'add_to_agent_memory'):
            memory_text = f"## My Opening Statement for Annual Review Year {year}\n\n{opening_statement}\n\n"
            self.document_manager.add_to_agent_memory(
                self.company.ceo.config.name,
                memory_text,
                tags=["annual_review", f"year_{year}", "opening_statement"]
            )
        if hasattr(self.company, 'decision_tracker'):
            self.company.decision_tracker.record_review_opening(meeting_id, opening_statement)
        return opening_statement

    def _collect_shareholder_submissions(self, meeting_id, ceo_statement):
        """Collects written evaluations from shareholders about CEO leadership and company performance in parallel."""

        # Get evaluation criteria for reference
        evaluation_criteria = self._define_evaluation_criteria()

        # Extract year from meeting_id
        year = int(meeting_id.split('_')[2].replace('year', ''))

        # Get or create year meeting summaries
        meeting_summaries = self._create_meeting_summaries_for_review(year)

        # Create evaluation dimensions text from criteria
        evaluation_dimensions = ""
        if evaluation_criteria:
            # CEO evaluation dimensions
            evaluation_dimensions += "When evaluating leadership, consider these key dimensions:\n"
            for key, criteria in evaluation_criteria["ceo"].items():
                evaluation_dimensions += f"- {criteria['title']}: {criteria['description']}\n"

            # Company evaluation dimensions
            evaluation_dimensions += "\nWhen evaluating company performance, consider these dimensions:\n"
            for key, criteria in evaluation_criteria["company"].items():
                evaluation_dimensions += f"- {criteria['title']}: {criteria['description']}\n"

        # Define worker function for parallel processing
        def _submission_worker(shareholder_info):
            shareholder_name, shareholder = shareholder_info

            # Get the shareholder's perspective context
            shareholder_context = ""
            if hasattr(self.document_manager, 'get_agent_context'):
                shareholder_context = self.document_manager.get_agent_context(shareholder_name)

            # Get position summaries from all quarterly meetings for this year
            quarterly_position_summaries = ""
            for q in range(1, 5):
                quarterly_meeting_id = f"quarterly_investment_year{year}_q{q}"
                try:
                    quarterly_summary = self.position_manager.get_position_summary_text(quarterly_meeting_id,
                                                                                        shareholder_name)
                    quarterly_position_summaries += f"## Q{q} Investment Position Summary:\n{quarterly_summary}\n\n"
                except:
                    # Skip if this quarterly meeting hasn't happened yet
                    pass

            # Get annual budget meeting position summary
            budget_meeting_id = f"annual_budget_year{year}"
            budget_position_summary = ""
            try:
                budget_position_summary = self.position_manager.get_position_summary_text(budget_meeting_id,
                                                                                          shareholder_name)
            except:
                pass

            # Add position summaries to the context if available
            position_context = ""
            if budget_position_summary:
                position_context += f"## Annual Budget Position Summary:\n{budget_position_summary}\n\n"
            if quarterly_position_summaries:
                position_context += quarterly_position_summaries

            spec = entity.free_action_spec(
                call_to_action=f"""You are {shareholder_name}, a shareholder in the investment company with MBTI personality type {shareholder.config.traits}.

    {self.SHAREHOLDER_DISCUSSION_GOAL}

    The CEO has presented the following annual review statement:
    "{ceo_statement}"

    Here is a summary of this year's meetings and performance:
    {meeting_summaries["overall_summary"]}

    {evaluation_dimensions}

    {position_context}

    IMPORTANT: Write your evaluation in formal first person ("I/my") as a professional written document. Express your assessment while maintaining consistent perspective.
    Based on your MBTI personality type and the company's performance this year, provide your assessment focusing on SPECIFIC EXAMPLES rather than general impressions:

    1. CEO STRENGTHS (2-3 specific examples of what the CEO did well)
    2. CEO AREAS FOR IMPROVEMENT (2-3 specific examples of how the CEO could improve)
    3. COMPANY STRENGTHS (2-3 specific examples of what went well with company performance)
    4. COMPANY CONCERNS (2-3 specific examples of company performance issues)
    5. QUESTIONS FOR CEO (1-2 direct questions you want the CEO to address about their leadership or company decisions)

    Your submission should be concrete, referencing specific decisions, meetings, or events from this year. Keep each point brief (1-2 sentences).

    Your response MUST use this exact format:
    CEO STRENGTHS:
    - [Specific example 1]
    - [Specific example 2]

    CEO AREAS FOR IMPROVEMENT:
    - [Specific example 1]
    - [Specific example 2]

    COMPANY STRENGTHS:
    - [Specific example 1]
    - [Specific example 2]

    COMPANY CONCERNS:
    - [Specific example 1]
    - [Specific example 2]

    QUESTIONS FOR CEO:
    1. [Specific question 1]
    2. [Specific question 2]

    IMPORTANT FORMAT INSTRUCTION: Your response MUST strictly follow the exact output format specified above. Do not add any explanations, notes, or content outside this format. Any deviation from the required format structure will cause processing errors. Provide ONLY the formatted response as outlined.

    ===MEETING CONTEXT===
    {shareholder_context}
    =====================""",
                tag=f"{meeting_id}_submission"
            )

            submission = shareholder.agent.act(spec)

            # Extract questions from submission
            import re
            questions = []
            questions_section = re.search(r'QUESTIONS FOR CEO:(.*?)(?:$|===)', submission, re.DOTALL | re.IGNORECASE)
            if questions_section:
                # Extract individual questions
                question_text = questions_section.group(1).strip()
                question_items = re.findall(r'\d+\.\s+(.*?)(?=\d+\.|$)', question_text + '\n0. ', re.DOTALL)
                questions = [q.strip() for q in question_items if q.strip()]

            # Store submission in shareholder's memory document
            if hasattr(self.document_manager, 'add_to_agent_memory'):
                memory_text = f"## My Annual Review Submission for Year {year}\n\n{submission}\n\n"
                self.document_manager.add_to_agent_memory(
                    shareholder_name,
                    memory_text,
                    tags=["annual_review", f"year_{year}", "my_submission"]
                )
            if hasattr(self.company, 'decision_tracker'):
                self.company.decision_tracker.record_shareholder_review(
                    meeting_id, shareholder_name, submission, None  # ratings will be recorded separately
                )
            # Return all needed data
            return shareholder_name, submission, questions

        # Process all shareholders in parallel
        submissions = {}
        questions = {}
        shareholder_items = list(self.company.shareholders.items())

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(shareholder_items)) as executor:
            for shareholder_name, submission, shareholder_questions in executor.map(_submission_worker,
                                                                                    shareholder_items):
                submissions[shareholder_name] = submission
                questions[shareholder_name] = shareholder_questions

                # Announce submission
                submission_announcement = f"{shareholder_name} has submitted their written evaluation for the annual review."
                for sh in self.company.shareholders.values():
                    sh.agent.observe(submission_announcement)
                self.company.ceo.agent.observe(submission_announcement)

        return {"submissions": submissions, "questions": questions}

    def _get_ceo_response_to_submissions(self, meeting_id, shareholder_data):
        """Helps CEO formulate a response to shareholder submissions and questions."""

        # Get the CEO's perspective context
        ceo_context = ""
        if hasattr(self.document_manager, 'get_agent_context'):
            ceo_context = self.document_manager.get_agent_context(self.company.ceo.config.name)

        # Extract submissions and questions
        submissions = shareholder_data.get("submissions", {})
        questions = shareholder_data.get("questions", {})

        # Get evaluation criteria for reference
        evaluation_criteria = self._define_evaluation_criteria()

        # Extract year from meeting_id
        year = int(meeting_id.split('_')[2].replace('year', ''))

        # Format submissions for CEO review
        submissions_text = "Shareholder Feedback Summary:\n\n"
        for name, submission in submissions.items():
            submissions_text += f"===== {name}'s Submission =====\n{submission}\n\n"

        # Organize questions by shareholder
        questions_text = "Shareholder Questions:\n\n"
        for name, shareholder_questions in questions.items():
            if shareholder_questions:
                questions_text += f"From {name}:\n"
                for i, question in enumerate(shareholder_questions, 1):
                    questions_text += f"{i}. {question}\n"
                questions_text += "\n"

        # If there are no questions
        if all(len(qs) == 0 for qs in questions.values()):
            questions_text += "No specific questions were submitted by shareholders.\n\n"

        # Create CEO evaluation categories reference
        ceo_categories = "\n".join([f"- {cat['title']}: {cat['description']}"
                                    for key, cat in evaluation_criteria["ceo"].items()])

        # Create company evaluation categories reference
        company_categories = "\n".join([f"- {cat['title']}: {cat['description']}"
                                        for key, cat in evaluation_criteria["company"].items()])

        spec = entity.free_action_spec(
            call_to_action=f"""You are {self.company.ceo.config.name}, the CEO of the investment company with MBTI personality type {self.company.ceo.config.traits}.

{self.CEO_DISCUSSION_GOAL}

Your shareholders have submitted their annual reviews and you need to respond thoughtfully to their feedback and questions. This is a critical leadership moment where you demonstrate accountability, reflection, and vision.

Shareholder feedback:
{submissions_text}

{questions_text}

For your response, address the following:

1. FEEDBACK THEMES: Identify and address 2-3 common themes from shareholder feedback (both positive and critical)
2. QUESTION RESPONSES: Directly answer each specific question raised by shareholders
3. LEADERSHIP SELF-ASSESSMENT: Provide your own assessment of your leadership against these key dimensions:
{ceo_categories}
4. COMPANY PERFORMANCE ASSESSMENT: Provide your assessment of the company's performance against these key dimensions:
{company_categories}
5. FUTURE DIRECTION: Outline 2-3 specific improvements or changes you intend to make based on this feedback

IMPORTANT: Deliver your formal speech in first person ("I/my" and when appropriate "we/our"). Never refer to yourself by name or in third person.
Your response should be thoughtful and specific, referencing concrete examples from the year. Demonstrate both confidence in your leadership and genuine openness to feedback.

Respond in a way that authentically reflects your MBTI personality type - whether that means being analytical, empathetic, decisive, or visionary.

===MEETING CONTEXT===
    {ceo_context}
=====================""",
            tag=f"{meeting_id}_ceo_response"
        )

        response = self.company.ceo.agent.act(spec)

        # Add response to CEO's memory document
        if hasattr(self.document_manager, 'add_to_agent_memory'):
            memory_text = f"## CEO Response to Annual Review Submissions for Year {year}\n\n{response}\n\n"
            self.document_manager.add_to_agent_memory(
                self.company.ceo.config.name,
                memory_text,
                tags=["annual_review", f"year_{year}", "ceo_response"]
            )
        if hasattr(self.company, 'decision_tracker'):
            self.company.decision_tracker.record_review_response(meeting_id, response)
        # Notify all participants
        for shareholder in self.company.shareholders.values():
            shareholder.agent.observe(response)

        self.company.ceo.agent.observe(response)

        return response

    def _collect_shareholder_ratings(self, meeting_id):
        """Collects numerical ratings from shareholders about CEO leadership and company performance in parallel."""

        # Get evaluation criteria
        evaluation_criteria = self._define_evaluation_criteria()

        # Extract year from meeting_id
        year = int(meeting_id.split('_')[2].replace('year', ''))

        # Initialize results structures
        ratings = {
            "ceo": {dimension: {} for dimension in evaluation_criteria["ceo"].keys()},
            "company": {dimension: {} for dimension in evaluation_criteria["company"].keys()},
            "averages": {
                "ceo": {},
                "company": {}
            }
        }

        # Format evaluation criteria for the prompt
        ceo_criteria_text = ""
        for key, criteria in evaluation_criteria["ceo"].items():
            ceo_criteria_text += f"### {criteria['title']}\n"
            ceo_criteria_text += f"{criteria['description']}\n"
            ceo_criteria_text += f"Question: {criteria['question']}\n"
            ceo_criteria_text += "Rating scale:\n"
            for rating, description in criteria["scale"].items():
                ceo_criteria_text += f"- {rating}: {description}\n"
            ceo_criteria_text += "\n"

        company_criteria_text = ""
        for key, criteria in evaluation_criteria["company"].items():
            company_criteria_text += f"### {criteria['title']}\n"
            company_criteria_text += f"{criteria['description']}\n"
            company_criteria_text += f"Question: {criteria['question']}\n"
            company_criteria_text += "Rating scale:\n"
            for rating, description in criteria["scale"].items():
                company_criteria_text += f"- {rating}: {description}\n"
            company_criteria_text += "\n"

        # Worker function to process a single shareholder
        def process_shareholder_rating(shareholder_data):
            shareholder_name, shareholder = shareholder_data

            # Get shareholder's context
            shareholder_context = ""
            if hasattr(self.document_manager, 'get_agent_context'):
                shareholder_context = self.document_manager.get_agent_context(shareholder_name)

            # Get position summaries from all quarterly meetings for this year
            quarterly_position_summaries = ""
            for q in range(1, 5):
                quarterly_meeting_id = f"quarterly_investment_year{year}_q{q}"
                try:
                    quarterly_summary = self.position_manager.get_position_summary_text(quarterly_meeting_id,
                                                                                        shareholder_name)
                    quarterly_position_summaries += f"## Q{q} Investment Position Summary:\n{quarterly_summary}\n\n"
                except:
                    # Skip if this quarterly meeting hasn't happened yet
                    pass

            # Get annual budget meeting position summary
            budget_meeting_id = f"annual_budget_year{year}"
            budget_position_summary = ""
            try:
                budget_position_summary = self.position_manager.get_position_summary_text(budget_meeting_id,
                                                                                          shareholder_name)
            except:
                pass

            # Add position summaries to the context if available
            position_context = ""
            if budget_position_summary:
                position_context += f"## Annual Budget Position Summary:\n{budget_position_summary}\n\n"
            if quarterly_position_summaries:
                position_context += quarterly_position_summaries

            spec = entity.free_action_spec(
                call_to_action=f"""You are {shareholder_name}, a shareholder in the investment company with MBTI personality type {shareholder.config.traits}.

    {self.SHAREHOLDER_DISCUSSION_GOAL}

    After reviewing the company's performance and the CEO's leadership this year, and hearing the CEO's response to feedback, you will now provide detailed ratings across multiple dimensions.

    {position_context}

    ## CEO EVALUATION CRITERIA:
    {ceo_criteria_text}

    ## COMPANY EVALUATION CRITERIA:
    {company_criteria_text}

    Based on your MBTI personality type and your experience with the company this year, provide ratings for each dimension on a 1-5 scale. For each rating, provide a brief 1-sentence explanation that references specific examples from this year.
    IMPORTANT: Address all meeting participants in first person ("I/my"). Never refer to yourself by name or in third person. Maintain a professional tone suitable for addressing the entire group.
    Your response MUST use this exact format:

    CEO RATINGS:
    LEADERSHIP_EFFECTIVENESS: [1-5] - [Brief explanation]
    DECISION_QUALITY: [1-5] - [Brief explanation]
    COMMUNICATION_CLARITY: [1-5] - [Brief explanation]
    RESPONSIVENESS_TO_INPUT: [1-5] - [Brief explanation]
    ADAPTABILITY: [1-5] - [Brief explanation]

    COMPANY RATINGS:
    FINANCIAL_RESULTS: [1-5] - [Brief explanation]
    RISK_MANAGEMENT: [1-5] - [Brief explanation]
    STRATEGIC_ALIGNMENT: [1-5] - [Brief explanation]
    CAPITAL_UTILIZATION: [1-5] - [Brief explanation]
    MARKET_RESPONSIVENESS: [1-5] - [Brief explanation]

    IMPORTANT FORMAT INSTRUCTION: Your response MUST strictly follow the exact output format specified above. Do not add any explanations, notes, or content outside this format. Any deviation from the required format structure will cause processing errors. Provide ONLY the formatted response as outlined.

    ===MEETING CONTEXT===
        {shareholder_context}
    =====================""",
                tag=f"{meeting_id}_ratings"
            )

            rating_response = shareholder.agent.act(spec)

            # Store ratings in shareholder's memory document
            if hasattr(self.document_manager, 'add_to_agent_memory'):
                memory_text = f"## My Ratings for Annual Review Year {year}\n\n{rating_response}\n\n"
                self.document_manager.add_to_agent_memory(
                    shareholder_name,
                    memory_text,
                    tags=["annual_review", f"year_{year}", "my_ratings"]
                )

            # Parse the ratings using regex
            import re

            # Initialize local rating results for this shareholder
            shareholder_ratings = {
                "ceo": {},
                "company": {}
            }

            # Extract CEO ratings
            for dimension in evaluation_criteria["ceo"].keys():
                pattern = f"{dimension.upper()}:\\s*(\\d)\\s*-\\s*(.*?)(?=\\n|$)"
                match = re.search(pattern, rating_response, re.IGNORECASE)
                if match:
                    score = int(match.group(1))
                    explanation = match.group(2).strip()
                    shareholder_ratings["ceo"][dimension] = {
                        "score": score,
                        "explanation": explanation
                    }
                else:
                    # Default value if parsing fails
                    shareholder_ratings["ceo"][dimension] = {
                        "score": 3,
                        "explanation": "No specific feedback provided"
                    }

            # Extract company ratings
            for dimension in evaluation_criteria["company"].keys():
                pattern = f"{dimension.upper()}:\\s*(\\d)\\s*-\\s*(.*?)(?=\\n|$)"
                match = re.search(pattern, rating_response, re.IGNORECASE)
                if match:
                    score = int(match.group(1))
                    explanation = match.group(2).strip()
                    shareholder_ratings["company"][dimension] = {
                        "score": score,
                        "explanation": explanation
                    }
                else:
                    # Default value if parsing fails
                    shareholder_ratings["company"][dimension] = {
                        "score": 3,
                        "explanation": "No specific feedback provided"
                    }
            # Record shareholder's ratings
            if hasattr(self.company, 'decision_tracker'):
                # Extract CEO rating and Company rating from shareholder_ratings
                ceo_overall = sum(item["score"] for item in shareholder_ratings["ceo"].values()) / len(
                    shareholder_ratings["ceo"]) if shareholder_ratings["ceo"] else 0
                company_overall = sum(item["score"] for item in shareholder_ratings["company"].values()) / len(
                    shareholder_ratings["company"]) if shareholder_ratings["company"] else 0

                # Update the previous submission with ratings
                self.company.decision_tracker.record_shareholder_review(
                    meeting_id,
                    shareholder_name,
                    None,  # submission was already recorded
                    {
                        "ceo_rating": int(ceo_overall),
                        "company_rating": int(company_overall),
                        "detailed_ratings": shareholder_ratings
                    }
                )
            # Announce ratings submission
            rating_announcement = f"{shareholder_name} has submitted their ratings for the annual review."
            for sh in self.company.shareholders.values():
                sh.agent.observe(rating_announcement)
            self.company.ceo.agent.observe(rating_announcement)

            return shareholder_name, shareholder_ratings

        # Use ThreadPoolExecutor to process all shareholders in parallel
        shareholders_data = list(self.company.shareholders.items())

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(shareholders_data)) as executor:
            for shareholder_name, shareholder_ratings in executor.map(process_shareholder_rating, shareholders_data):
                # Merge results back into the main ratings structure
                for entity_type in ["ceo", "company"]:
                    for dimension in evaluation_criteria[entity_type].keys():
                        if dimension in shareholder_ratings[entity_type]:
                            ratings[entity_type][dimension][shareholder_name] = shareholder_ratings[entity_type][
                                dimension]

        # Calculate average scores for each dimension
        for entity_type in ["ceo", "company"]:
            for dimension in evaluation_criteria[entity_type].keys():
                scores = [data["score"] for shareholder, data in ratings[entity_type][dimension].items()]
                if scores:
                    ratings["averages"][entity_type][dimension] = sum(scores) / len(scores)
                else:
                    ratings["averages"][entity_type][dimension] = 0

        # Calculate overall averages
        if ratings["averages"]["ceo"]:
            ratings["averages"]["ceo"]["overall"] = sum(ratings["averages"]["ceo"].values()) / len(
                ratings["averages"]["ceo"])
        else:
            ratings["averages"]["ceo"]["overall"] = 0

        if ratings["averages"]["company"]:
            ratings["averages"]["company"]["overall"] = sum(ratings["averages"]["company"].values()) / len(
                ratings["averages"]["company"])
        else:
            ratings["averages"]["company"]["overall"] = 0

        # Format and display rating results
        rating_results = f"# Annual Review Rating Results\n\n"

        # CEO ratings
        rating_results += "## CEO Performance Ratings\n\n"
        for dimension, avg_score in ratings["averages"]["ceo"].items():
            if dimension != "overall":
                dimension_title = evaluation_criteria["ceo"].get(dimension, {}).get("title", dimension.capitalize())
                rating_results += f"### {dimension_title}: {avg_score:.1f}/5\n\n"
                for shareholder, data in ratings["ceo"][dimension].items():
                    rating_results += f"- {shareholder}: {data['score']}/5 - {data['explanation']}\n"
                rating_results += "\n"

        rating_results += f"### Overall CEO Rating: {ratings['averages']['ceo']['overall']:.1f}/5\n\n"

        # Company ratings
        rating_results += "## Company Performance Ratings\n\n"
        for dimension, avg_score in ratings["averages"]["company"].items():
            if dimension != "overall":
                dimension_title = evaluation_criteria["company"].get(dimension, {}).get("title", dimension.capitalize())
                rating_results += f"### {dimension_title}: {avg_score:.1f}/5\n\n"
                for shareholder, data in ratings["company"][dimension].items():
                    rating_results += f"- {shareholder}: {data['score']}/5 - {data['explanation']}\n"
                rating_results += "\n"

        rating_results += f"### Overall Company Rating: {ratings['averages']['company']['overall']:.1f}/5\n\n"

        # Add ratings summary to all documents
        if hasattr(self.document_manager, 'add_to_all_documents'):
            self.document_manager.add_to_all_documents(rating_results, tags=["ratings_results"])

        # Add summary to each agent's memory
        if hasattr(self.document_manager, 'add_to_agent_memory'):
            # Create a simplified rating summary for memory
            memory_summary = f"## Annual Review Ratings for Year {year}\n\n"
            memory_summary += f"CEO rating: {ratings['averages']['ceo']['overall']:.1f}/10\n"
            memory_summary += f"Company rating: {ratings['averages']['company']['overall']:.1f}/10\n\n"

            # Add detailed ratings for each dimension
            memory_summary += "### CEO Performance Ratings\n"
            for dimension, score in ratings["averages"]["ceo"].items():
                if dimension != "overall":
                    dimension_title = evaluation_criteria["ceo"].get(dimension, {}).get("title", dimension.capitalize())
                    memory_summary += f"- {dimension_title}: {score:.1f}/5\n"

            memory_summary += "\n### Company Performance Ratings\n"
            for dimension, score in ratings["averages"]["company"].items():
                if dimension != "overall":
                    dimension_title = evaluation_criteria["company"].get(dimension, {}).get("title",
                                                                                            dimension.capitalize())
                    memory_summary += f"- {dimension_title}: {score:.1f}/5\n"

            # Add to all agents' memory documents - FIXED THIS LINE
            for agent_name in [self.company.ceo.config.name] + list(self.company.shareholders.keys()):
                self.document_manager.add_to_agent_memory(
                    agent_name,
                    memory_summary,
                    tags=["annual_review", f"year_{year}", "ratings_summary"]
                )

        # Share results with all participants
        for shareholder in self.company.shareholders.values():
            shareholder.agent.observe(rating_results)
        self.company.ceo.agent.observe(rating_results)
        # Record company review results in decision tracker
        if hasattr(self.company, 'decision_tracker'):
            self.company.decision_tracker.record_company_review(
                meeting_id,
                {
                    "average_score": ratings["averages"]["company"]["overall"],
                    "detailed_scores": ratings["company"],
                    "individual_ratings": {name: data for name, data in ratings["company"].items()}
                },
                {
                    "average_score": ratings["averages"]["ceo"]["overall"],
                    "detailed_scores": ratings["ceo"],
                    "individual_ratings": {name: data for name, data in ratings["ceo"].items()}
                }
            )
        return ratings

    def _get_ceo_closing_statement_for_review(self, meeting_id, ratings_data):
        """Helps CEO craft a closing statement for the annual review meeting."""

        # Get the CEO's perspective context
        ceo_context = ""
        if hasattr(self.document_manager, 'get_agent_context'):
            ceo_context = self.document_manager.get_agent_context(self.company.ceo.config.name)

        # Extract year from meeting_id
        year = int(meeting_id.split('_')[2].replace('year', ''))

        # Extract key information from ratings
        ceo_overall = ratings_data["averages"]["ceo"]["overall"]
        company_overall = ratings_data["averages"]["company"]["overall"]

        # Identify highest and lowest rated dimensions
        ceo_dimensions = {k: v for k, v in ratings_data["averages"]["ceo"].items() if k != "overall"}
        company_dimensions = {k: v for k, v in ratings_data["averages"]["company"].items() if k != "overall"}

        ceo_strength = max(ceo_dimensions.items(), key=lambda x: x[1]) if ceo_dimensions else ("unknown", 0)
        ceo_challenge = min(ceo_dimensions.items(), key=lambda x: x[1]) if ceo_dimensions else ("unknown", 0)

        company_strength = max(company_dimensions.items(), key=lambda x: x[1]) if company_dimensions else ("unknown", 0)
        company_challenge = min(company_dimensions.items(), key=lambda x: x[1]) if company_dimensions else ("unknown",
                                                                                                            0)

        # Format dimension names for readability
        ceo_strength_name = ceo_strength[0].replace("_", " ").title()
        ceo_challenge_name = ceo_challenge[0].replace("_", " ").title()
        company_strength_name = company_strength[0].replace("_", " ").title()
        company_challenge_name = company_challenge[0].replace("_", " ").title()

        spec = entity.free_action_spec(
            call_to_action=f"""You are {self.company.ceo.config.name}, the CEO of the investment company with MBTI personality type {self.company.ceo.config.traits}. 

{self.CEO_DISCUSSION_GOAL}

You're concluding the annual review meeting. The shareholders have rated your leadership and company performance across multiple dimensions:

CEO OVERALL RATING: {ceo_overall:.1f}/5
- Highest rated dimension: {ceo_strength_name} ({ceo_strength[1]:.1f}/5)
- Lowest rated dimension: {ceo_challenge_name} ({ceo_challenge[1]:.1f}/5)

COMPANY OVERALL RATING: {company_overall:.1f}/5
- Highest rated dimension: {company_strength_name} ({company_strength[1]:.1f}/5)
- Lowest rated dimension: {company_challenge_name} ({company_challenge[1]:.1f}/5)

This is your opportunity to provide a meaningful closing statement that:
1. Acknowledges the ratings in a way authentic to your MBTI personality type
2. Addresses both strengths and areas for improvement
3. Outlines 2-3 specific commitments for improvement in the coming year
4. Ends with a forward-looking vision that inspires confidence

IMPORTANT: Deliver your formal speech in first person ("I/my" and when appropriate "we/our"). Never refer to yourself by name or in third person.
Keep your closing statement concise (4-6 sentences) and focused on what these ratings mean for your leadership and the company's future.

===MEETING CONTEXT===
    {ceo_context}
=====================""",
            tag=f"{meeting_id}_closing"
        )

        closing_statement = self.company.ceo.agent.act(spec)

        # Add closing statement to CEO's memory document
        if hasattr(self.document_manager, 'add_to_agent_memory'):
            memory_text = f"## My Closing Statement for Annual Review Year {year}\n\n{closing_statement}\n\n"
            self.document_manager.add_to_agent_memory(
                self.company.ceo.config.name,
                memory_text,
                tags=["annual_review", f"year_{year}", "closing_statement"]
            )
        if hasattr(self.company, 'decision_tracker'):
            self.company.decision_tracker.record_review_closing(meeting_id, closing_statement)
        return closing_statement

    def _update_participants_with_round(self, meeting_id, round_log, discussion_log):
        """Update all participants after a round with position reassessment, reflection, and summaries."""
        # Get the current round number
        current_round = round_log.get('round', 0)

        # STEP 1: Generate round summaries for all participants
        if hasattr(self.document_manager, 'generate_round_summaries'):
            # Generate summaries for CEO
            ceo_name = self.company.ceo.config.name
            ceo_summaries = self.document_manager.generate_round_summaries(
                meeting_id, current_round, ceo_name
            )

            # Record CEO's personal summaries in DecisionTracker
            if hasattr(self.company, 'decision_tracker') and ceo_summaries:
                combined_summary = self._extract_combined_summary(ceo_summaries)
                self.company.decision_tracker.record_personal_summary(
                    agent_name=ceo_name,
                    meeting_id=meeting_id,
                    round_num=current_round,
                    summary=combined_summary
                )

            # Define worker function for parallel processing
            def _generate_summaries_worker(shareholder_info):
                shareholder_name, shareholder = shareholder_info
                shareholder_summaries = self.document_manager.generate_round_summaries(
                    meeting_id, current_round, shareholder_name
                )

                # Record personal summaries in DecisionTracker
                if hasattr(self.company, 'decision_tracker') and shareholder_summaries:
                    combined_summary = self._extract_combined_summary(shareholder_summaries)
                    self.company.decision_tracker.record_personal_summary(
                        agent_name=shareholder_name,
                        meeting_id=meeting_id,
                        round_num=current_round,
                        summary=combined_summary
                    )

                return shareholder_name, shareholder_summaries

            # Generate summaries for shareholders in parallel
            shareholder_items = list(self.company.shareholders.items())
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(shareholder_items)) as executor:
                for shareholder_name, shareholder_summaries in executor.map(_generate_summaries_worker,
                                                                            shareholder_items):
                    # If you need to do something with the results, you can add processing here
                    pass
            # time.sleep(5)
        # STEP 2: Generate reflections for agents
        # Extract year and quarter from meeting_id for tagging
        year = int(meeting_id.split('_')[2].replace('year', ''))
        quarter = None
        if 'q' in meeting_id:
            quarter = int(meeting_id.split('_')[-1].replace('q', ''))

        # Determine meeting type from meeting_id
        meeting_type = "meeting"
        if "annual_budget" in meeting_id:
            meeting_type = "annual_budget"
        elif "quarterly_investment" in meeting_id:
            meeting_type = "quarterly_investment"
        elif "annual_review" in meeting_id:
            meeting_type = "annual_review"

        # PARALLELIZED: Generate agent reflections for all agents in parallel
        def _generate_reflection_worker(agent):
            agent_reflection = self._generate_agent_reflection_after_round(
                agent,
                meeting_id,
                meeting_type,
                current_round,
                year,
                quarter
            )
            return agent.config.name, agent_reflection

        # Collect all agents (CEO + shareholders)
        all_agents = [self.company.ceo] + list(self.company.shareholders.values())

        # Process all agents in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(all_agents)) as executor:
            for agent_name, agent_reflection in executor.map(_generate_reflection_worker, all_agents):
                # If you need to do something with the reflections, you can process them here
                pass
        # time.sleep(5)
        # STEP 3: Trigger position reassessment for all participants
        # Get the options directly from the discussion_log parameter
        options = discussion_log.get('options', {})

        if not options:
            print(f"Warning: Could not find options for meeting {meeting_id}. Position reassessment will be skipped.")
            return  # Can't reassess without options

        # Reassess CEO position
        ceo_name = self.company.ceo.config.name
        if self.position_manager.get_current_position(meeting_id, ceo_name):
            updated_position, updated_reasoning, updated_confidence, position_changed = self._reassess_position_for_ceo(
                meeting_id,
                meeting_type,
                current_round,
                options,
                round_log
            )

        # Reassess shareholder positions in parallel
        shareholder_reassessments = self._parallel_reassess_positions(
            meeting_id,
            meeting_type,
            current_round,
            options,
            round_log
        )
        # time.sleep(5)
    def _generate_agent_reflection_after_round(self, agent, meeting_id, meeting_type, current_round, year,
                                               quarter=None):
        """Generates reflection for an agent after a discussion round."""

        # Get the agent's context for reflection
        agent_context = ""
        if hasattr(self.document_manager, 'get_agent_context'):
            agent_context = self.document_manager.get_agent_context(agent.config.name)

        # Create an interactive document for generating the reflection
        from concordia.document.interactive_document import InteractiveDocument
        reflection_doc = InteractiveDocument(model=self.model)

        # Extract the agent's personality type
        mbti_type = agent.config.traits if hasattr(agent.config, 'traits') else "Unknown"
        is_ceo = agent.config.name == self.company.ceo.config.name
        role = "CEO" if is_ceo else "Shareholder"

        # Get agent's position history if available
        position_history = None
        initial_position = None
        current_position = None

        if hasattr(self, 'position_manager'):
            if self.position_manager._check_position_exists(meeting_id, agent.config.name):
                position_history = self.position_manager.get_position_history(meeting_id, agent.config.name)
                initial_position = self.position_manager.get_initial_position(meeting_id, agent.config.name)
                current_position = self.position_manager.get_current_position(meeting_id, agent.config.name)

        # Create position evolution context if available
        position_context = ""
        if position_history and initial_position and current_position:
            position_context = f"""
    POSITION EVOLUTION CONTEXT:
    - Initial position: {initial_position.get('option', 'Unknown')}
    - Current position: {current_position.get('option', 'Unknown')}
    - Position changes: {len([p for p in position_history if p.get('changed', False)])}
    """

            # Add confidence evolution if available
            initial_confidence = initial_position.get('confidence_distribution', {})
            current_confidence = current_position.get('confidence_distribution', {})

            if initial_confidence and current_confidence:
                position_context += "\nCONFIDENCE EVOLUTION:\n"
                for option in initial_confidence.keys():
                    initial_value = initial_confidence.get(option, 0) * 100
                    current_value = current_confidence.get(option, 0) * 100
                    change = current_value - initial_value
                    position_context += f"- {option}: {initial_value:.1f}% → {current_value:.1f}% ({'+' if change > 0 else ''}{change:.1f}%)\n"

        # Create prompt for reflection generation
        reflection_prompt = f"""As {agent.config.name} with MBTI personality type {mbti_type} and role {role}, reflect on how your thinking has evolved through this discussion round.

{position_context}

IMPORTANT: Frame your thinking in first person ("I/my") as an internal reflection process. Express your analytical considerations and personal assessment with authentic introspection.

Consider:

1. INITIAL PERSPECTIVE: 
   - What was my initial position and reasoning?
   - What factors or assumptions were most important in my initial assessment?

2. KEY INFLUENCES:
   - Which specific points from others have most influenced my thinking?
   - What arguments were most compelling, regardless of source?
   - Which perspectives challenged my initial assumptions most effectively?

3. THOUGHT EVOLUTION:
   - How has my understanding of the issue deepened or changed?
   - What aspects of my assessment have been reinforced?
   - What aspects have been modified or refined?

4. CONFIDENCE SHIFTS:
   - How has my confidence in different options shifted and why?
   - What specific evidence or reasoning caused these shifts?
   - Are there options I'm viewing differently now than at the start?

5. INTEGRATION:
   - How have I incorporated others' insights into my own thinking?
   - What synthesis of different perspectives has emerged in my understanding?
   - What new considerations now seem important that I initially overlooked?

Focus on your authentic cognitive and evaluative process rather than strategic positioning. What would be most valuable to remember about how your thinking evolved in this discussion?
    """

        # Generate reflection using the agent's meeting document as context
        try:
            reflection = reflection_doc.open_question(
                reflection_prompt + "\n\n" + agent_context,
                answer_label="Reflection",
                max_tokens=500
            )
        except Exception as e:
            reflection = f"Unable to generate reflection due to error: {str(e)}"
        # Record the reflection in DecisionTracker
        if hasattr(self.company, 'decision_tracker'):
            self.company.decision_tracker.record_personal_reflection(
                agent_name=agent.config.name,
                meeting_id=meeting_id,
                round_num=current_round,
                reflection=reflection
            )
        # Format the reflection for storing in memory
        meeting_tag = meeting_type.replace("_", " ").title()
        quarter_tag = f", Quarter {quarter}" if quarter else ""

        reflection_memory = f"## My Reflection After Round {current_round} - {meeting_tag} Year {year}{quarter_tag}\n\n"
        reflection_memory += reflection + "\n\n"

        # Store the reflection in the agent's memory
        if hasattr(self.document_manager, 'add_to_agent_memory'):
            self.document_manager.add_to_agent_memory(
                agent.config.name,
                reflection_memory,
                tags=[meeting_type, f"year_{year}", f"round_{current_round}",
                      f"quarter_{quarter}" if quarter else "annual", "reflection"]
            )

        return reflection

    def _debug_voting_status(self, meeting_id):
        """Helper to verify voting status for a meeting."""
        if hasattr(self, 'voting_manager'):
            print(f"Meeting {meeting_id} voting status:")
            # print(f"  Has initial votes: {self.voting_manager.has_votes(meeting_id, 'initial')}")
            # print(f"  Has alternative votes: {self.voting_manager.has_votes(meeting_id, 'alternative')}")
            if self.voting_manager.has_votes(meeting_id, 'initial'):
                stats = self.voting_manager.get_vote_counts(meeting_id, 'initial')
                # print(f"  Initial vote approval: {stats['approve_percentage']:.1f}%")

    def _extract_combined_summary(self, summaries):
        """Extract the combined summary text from the summaries dictionary."""
        if not summaries:
            return ""

        # Combine all summary types into a single text
        combined = []
        if 'discussion_flow' in summaries:
            combined.append(f"Discussion Flow: {summaries['discussion_flow']}")
        if 'shareholder_positions' in summaries:
            combined.append(f"Shareholder Positions: {summaries['shareholder_positions']}")
        if 'decision_progress' in summaries:
            combined.append(f"Decision Progress: {summaries['decision_progress']}")
        if 'leadership_strategy' in summaries:
            combined.append(f"Leadership Strategy: {summaries['leadership_strategy']}")
        # For shareholders
        if 'self_dialogue' in summaries:
            combined.append(f"Self-CEO Dialogue: {summaries['self_dialogue']}")
        if 'others_contributions' in summaries:
            combined.append(f"Others' Contributions: {summaries['others_contributions']}")
        if 'positions_analysis' in summaries:
            combined.append(f"Positions Analysis: {summaries['positions_analysis']}")
        if 'ceo_analysis' in summaries:
            combined.append(f"CEO Analysis: {summaries['ceo_analysis']}")

        return "\n".join(combined) if combined else ""
# %% md
# # Simulation Class
# %%

class Simulation:
    """Controls the overall simulation flow and environment."""

    def __init__(self, agent_num, random_seed=None, model=None, result_dir="result", specific_mbti=None):
        # Set random seed if provided
        self.random_seed = random_seed
        if random_seed:
            random.seed(random_seed)
            np.random.seed(random_seed)

        self.agent_num = agent_num
        self.model = model or Model_And_Embedder.model
        self.run_years = None  # Will be set when run() is called
        self.specific_mbti = specific_mbti  # Store the specific MBTI

        # Setup simulation components
        self.profile = self.setup_profile(specific_mbti=specific_mbti)
        print('Setup profile')

        # Add this line to initialize the result saver
        self.result_saver = SimulationResultSaver(base_dir=result_dir)

        self.BackgroundInfo = BackgroundInfo(agent_num - 1, self.profile['ceo']['profile']['name'])
        print('Setup background info')

        with open(r"prompts.yaml", "r") as file:
            self.prompts = yaml.safe_load(file)

        self.shareholders = self.setup_shareholder()
        print('Setup shareholders')
        # time.sleep(10)
        self.ceo = CEO(self.profile['ceo'], self.BackgroundInfo, self.model)
        print('Setup CEO')

        # Create the simulation folder once we know the number of agents
        ceo_mbti=self.ceo.config.traits
        self.simulation_folder = self.result_saver.create_simulation_folder(agent_num,ceo_mbti)

        self.market = Market(seed=42)
        print('Setup market')

        self.company = Company(self.shareholders, self.ceo, self.market, self.prompts, model=self.model,simulation=self)
        print('Setup company')

        # Create a simulation document to record the entire simulation - now using InteractiveDocument
        self.simulation_document = InteractiveDocument(model=self.model)  # Use InteractiveDocument
        self.simulation_document.statement("# Investment Company Simulation\n\n")

        # Add this line to pass the background info to the company
        if hasattr(self.BackgroundInfo, 'market_condition_intro'):
            self.company.meeting_manager.market_condition_intros = self.BackgroundInfo.market_condition_intro

    def setup_profile(self, specific_mbti=None, team_composition=None):
        """Set up agent profiles including CEO and shareholders.

        Args:
            specific_mbti: Optional MBTI type to use for all agents (e.g., "INTJ")
            team_composition: Optional dict with "ceo" and "shareholders" keys specifying team composition
        """
        profile = dict()

        # If team composition is specified, use it
        if team_composition:
            ceo_mbti = team_composition["ceo"]
            shareholder_mbti = team_composition["shareholders"]

            # Create combined MBTI dict for all agents (CEO + shareholders)
            all_mbti = shareholder_mbti.copy()
            if ceo_mbti not in all_mbti:
                all_mbti[ceo_mbti] = 1
            else:
                all_mbti[ceo_mbti] += 1

            # Generate all profiles with their respective MBTI types
            profile['all'] = ChooseProfile(
                self.agent_num,
                mbti_specify=True,
                specify_mbti=all_mbti,
                mbti_diff=False,
                random_seed=self.random_seed
            )

            # Filter profiles by MBTI type
            ceo_candidates = [p for p in profile['all'] if p['mbti'] == ceo_mbti]
            if not ceo_candidates:
                raise ValueError(f"No profile with CEO MBTI type {ceo_mbti}")

            # Select CEO
            profile['ceo'] = random.choice(ceo_candidates)

            # Remove CEO from shareholders
            profile['shareholders'] = [p for p in profile['all'] if p != profile['ceo']]

            return profile

        # Original logic for when all agents have the same MBTI
        elif specific_mbti:
            # Verify the MBTI is valid
            valid_mbti_types = [
                "ISTJ", "ISFJ", "INFJ", "INTJ",
                "ISTP", "ISFP", "INFP", "INTP",
                "ESTP", "ESFP", "ENFP", "ENTP",
                "ESTJ", "ESFJ", "ENFJ", "ENTJ"
            ]

            if specific_mbti not in valid_mbti_types:
                raise ValueError(f"Invalid MBTI type: {specific_mbti}. Must be one of {valid_mbti_types}")

            # Use the same MBTI for all agents
            profile['all'] = ChooseProfile(
                self.agent_num,
                mbti_specify=True,
                specify_mbti=[specific_mbti],
                mbti_diff=False,
                random_seed=self.random_seed
            )
        else:
            # Original behavior - different MBTI types
            profile['all'] = ChooseProfile(self.agent_num, mbti_diff=True, random_seed=self.random_seed)

        profile['shareholders'] = profile['all'].copy()

        # Select CEO from shareholders (they all have the same MBTI if specific_mbti is set)
        profile['ceo'] = SetupLeader(profile['shareholders'], random_seed=self.random_seed)
        profile['shareholders'].remove(profile['ceo'])

        return profile

    def setup_shareholder(self):
        """Set up shareholder agents with concurrent processing."""

        def create_shareholder(config, bg_info, model):
            return Shareholder(config, bg_info, model)

        shareholders = dict()
        loop_iter = len(self.profile['shareholders'])

        with concurrent.futures.ThreadPoolExecutor(max_workers=loop_iter) as pool:
            for agent in pool.map(create_shareholder,
                                  self.profile['shareholders'],
                                  [self.BackgroundInfo] * loop_iter,
                                  [self.model] * loop_iter):
                shareholders[agent.config.name] = agent

        return shareholders

    def run(self, years=5):
        self.run_years = years

        # Record simulation start information in the document
        start_time = datetime.datetime.now()

        simulation_header = f"# Investment Company Simulation\n\n"
        simulation_header += f"Simulation started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        simulation_header += f"## Simulation Setup\n\n"
        simulation_header += f"- Number of agents: {self.agent_num}\n"
        simulation_header += f"- CEO: {self.ceo.config.name} (MBTI: {self.ceo.config.traits})\n"
        simulation_header += f"- Simulation period: {years} years (1971-{1971 + years - 1})\n\n"

        simulation_header += f"## Shareholders\n\n"
        for name, shareholder in self.shareholders.items():
            simulation_header += f"- {name} (MBTI: {shareholder.config.traits})\n"

        simulation_header += f"\n## Market Conditions\n\n"
        for year in range(1, years + 1):
            market_condition = self.market.market_conditions[year - 1]
            market_intro = self.BackgroundInfo.market_condition_intro.get(market_condition, "")
            simulation_header += f"- Year {year} ({1970 + year}): {market_condition.replace('_', ' ').title()}\n"

        self.simulation_document.statement(simulation_header)

        # Add the simulation document to the company's document manager if available
        if hasattr(self.company.meeting_manager, 'document_manager'):
            simulation_doc_text = self.simulation_document.text()
            self.company.meeting_manager.document_manager.add_to_all_documents(simulation_doc_text)

        # Print simulation start information
        print(f"\n=== Starting simulation with {self.agent_num} agents for {years} years (1971-{1971 + years - 1}) ===")
        print(f"CEO: {self.ceo.config.name} (MBTI: {self.ceo.config.traits})")
        print("Shareholders:")
        for name, shareholder in self.shareholders.items():
            print(f"- {name} (MBTI: {shareholder.config.traits})")

        results = {
            "yearly_results": [],
            "assets_history": [self.company.assets],
            "investment_decisions": [],
            "ceo_ratings": [],
            "company_ratings": [],
            "agent_reflections": {},  # NEW: Track agent reflections
            "meta": {
                "num_agents": self.agent_num,
                "ceo": {
                    "name": self.ceo.config.name,
                    "mbti": self.ceo.config.traits
                },
                "shareholders": [{
                    "name": shareholder.config.name,
                    "mbti": shareholder.config.traits
                } for shareholder in self.shareholders.values()]
            }
        }

        # Run each year of the simulation
        for year in range(1, years + 1):
            print(f"\n\n=== YEAR {year} ({str(int(1970) + int(year))}) ===")

            # Record year start in the document
            year_start_text = f"# Year {year} ({1970 + year})\n\n"
            year_start_text += f"Market condition: {self.market.market_conditions[year - 1]}\n"
            year_start_text += f"Starting assets: ${self.company.assets:.2f}\n\n"
            self.simulation_document.statement(year_start_text)

            year_results = self.company.run_annual_cycle(year)

            results["yearly_results"].append(year_results)
            results["assets_history"].extend(self.company.assets_logs[-4:])
            results["investment_decisions"].extend(
                [inv for inv in self.company.investment_decisions if inv["year"] == year])

            if "ceo_rating" in year_results:
                results["ceo_ratings"].append(year_results["ceo_rating"])
                results["company_ratings"].append(year_results["company_rating"])

            # Extract agent reflections (NEW)
            if hasattr(self.company.meeting_manager, 'document_manager'):
                doc_manager = self.company.meeting_manager.document_manager

                # Extract reflections for each agent
                for agent_name in [self.ceo.config.name] + list(self.shareholders.keys()):
                    agent_memory = doc_manager.get_agent_memory(agent_name)

                    # Simple extraction of year-related reflections (can be improved with regex)
                    year_tag = f"_year{year}_"
                    reflection_sections = []

                    # Find all reflections for this year
                    pos = 0
                    while True:
                        reflection_start = agent_memory.find("## Reflection:", pos)
                        if reflection_start == -1:
                            break

                        # Check if this reflection is for this year
                        reflection_id_start = reflection_start + 13  # Length of "## Reflection: "
                        reflection_id_end = agent_memory.find("\n", reflection_id_start)
                        if reflection_id_end == -1:
                            reflection_id_end = len(agent_memory)

                        reflection_id = agent_memory[reflection_id_start:reflection_id_end].strip()

                        if year_tag in reflection_id:
                            # Find the end of this reflection
                            next_section = agent_memory.find("##", reflection_start + 1)
                            if next_section == -1:
                                next_section = len(agent_memory)

                            reflection_content = agent_memory[reflection_start:next_section].strip()
                            reflection_sections.append({
                                "meeting_id": reflection_id,
                                "content": reflection_content
                            })

                        pos = reflection_start + 1

                    # Store reflections in results
                    if reflection_sections:
                        if "agent_reflections" not in results:
                            results["agent_reflections"] = {}

                        if agent_name not in results["agent_reflections"]:
                            results["agent_reflections"][agent_name] = []

                        results["agent_reflections"][agent_name].extend(reflection_sections)

            # Record year end in the document
            year_end_text = f"## Year {year} Results\n\n"
            year_end_text += f"Ending assets: ${self.company.assets:.2f}\n"
            year_end_text += f"Growth: {year_results['growth_percentage']:.2f}%\n"
            if "ceo_rating" in year_results:
                year_end_text += f"CEO rating: {year_results['ceo_rating']:.1f}/10\n"
                year_end_text += f"Company rating: {year_results['company_rating']:.1f}/10\n"
            year_end_text += "\n"
            self.simulation_document.statement(year_end_text)

        # Add final results
        results["final_assets"] = self.company.assets
        results["growth_percentage"] = ((self.company.assets / 100000) - 1) * 100

        # Record simulation final results
        simulation_summary = f"# Simulation Summary\n\n"
        simulation_summary += f"Initial assets: $100,000.00\n"
        simulation_summary += f"Final assets: ${results['final_assets']:.2f}\n"
        simulation_summary += f"Overall growth: {results['growth_percentage']:.2f}%\n\n"

        if results['ceo_ratings']:
            simulation_summary += f"## CEO Performance\n\n"
            simulation_summary += f"Average CEO rating: {sum(results['ceo_ratings']) / len(results['ceo_ratings']):.1f}/10\n\n"

        # NEW: Add agent reflection summary if available
        if "agent_reflections" in results and results["agent_reflections"]:
            simulation_summary += f"## Agent Reflection Highlights\n\n"

            for agent_name, reflections in results["agent_reflections"].items():
                if reflections:
                    simulation_summary += f"### {agent_name}'s Key Reflections\n\n"

                    # Extract meeting IDs for readability
                    meeting_types = set()
                    for reflection in reflections:
                        meeting_id = reflection.get("meeting_id", "")
                        if "_budget_" in meeting_id:
                            meeting_types.add("Budget Meetings")
                        elif "_investment_" in meeting_id:
                            meeting_types.add("Investment Meetings")
                        elif "_review_" in meeting_id:
                            meeting_types.add("Review Meetings")

                    # List meeting types this agent reflected on
                    simulation_summary += f"Reflected on: {', '.join(sorted(meeting_types))}\n"
                    simulation_summary += f"Total reflections: {len(reflections)}\n\n"

        simulation_summary += f"Simulation completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        self.simulation_document.statement(simulation_summary)

        # Add the final simulation document to the company's document manager if available
        if hasattr(self.company.meeting_manager, 'document_manager'):
            self.company.meeting_manager.document_manager.add_to_all_documents(simulation_summary)

        # NEW CODE: Compile final data into DecisionTracker
        print("\nCompiling final decision tracking data...")

        # Record simulation metadata
        if disable_language_model==True:
            models_info = {
                "main": None,
                "low": None,
                "high": None
            }
        else:
            models_info = {
                "main": model._model_name if 'model' in globals() else "unknown",
                "low": model_low._model_name if 'model_low' in globals() else "unknown",
                "high": model_high._model_name if 'model_high' in globals() else "unknown"
            }

        self.company.decision_tracker.record_simulation_metadata(
            years=years,
            random_seed=self.random_seed,
            market_conditions=self.market.market_conditions,
            models=models_info
        )

        # Record market returns data
        for condition in self.market.market_conditions:
            quarterly_returns = {}
            for asset in self.market.assets:
                # Get all returns for this market condition
                returns_for_condition = []
                for quarter in range(20):  # 5 years * 4 quarters
                    if self.market.get_market_condition_for_quarter(quarter) == condition:
                        returns_for_condition.append(self.market.returns_history[asset][quarter])
                if returns_for_condition:
                    quarterly_returns[asset] = returns_for_condition

            if quarterly_returns:
                self.company.decision_tracker.record_market_returns(condition, quarterly_returns)

        # Copy data from PositionManager
        if hasattr(self.company.meeting_manager, 'position_manager'):
            positions_data = self.company.meeting_manager.position_manager.positions
            self.company.decision_tracker.data["position"]["meetings"] = positions_data.copy()

        # Copy data from VotingManager
        if hasattr(self.company.meeting_manager, 'voting_manager'):
            voting_data = self.company.meeting_manager.voting_manager.votes
            self.company.decision_tracker.data["voting"] = voting_data.copy()
        ceo_mbti=self.ceo.config.traits
        # Export complete data structure to JSON
        import os
        decision_tracker_filename = os.path.join(self.simulation_folder, f"decision_tracker_data_{ceo_mbti}.json")
        self.company.decision_tracker.export_to_json(decision_tracker_filename)

        # Generate and print summary statistics
        summary_stats = self.company.decision_tracker.get_summary_statistics()
        print("\nDecision Tracker Summary:")
        print(f"Total meetings: {summary_stats['total_meetings']}")
        print(f"Total rounds: {summary_stats['total_rounds']}")
        print(f"Total votes: {summary_stats['total_votes']}")
        growth_rate = summary_stats['performance_metrics']['growth_rate']
        if growth_rate is not None:
            print(f"Growth rate: {growth_rate:.2f}%")
        else:
            print("Growth rate: Not available (final assets not recorded)")

        # Validate data integrity (optional but recommended)
        validation_results = self.company.decision_tracker.validate_data_integrity()
        if not validation_results["valid"]:
            print("\nData validation issues found:")
            for issue in validation_results["issues"]:
                print(f"- {issue}")

        print(f"Decision tracking data saved to: {decision_tracker_filename}")

        # Save all simulation data
        print("\nSaving all simulation data...")
        self.result_saver.save_simulation_results(self.simulation_folder, self.company)
        self.result_saver.save_agent_components(self.simulation_folder, self.shareholders, self.ceo)
        # ADD THIS LINE to save decision tracker separately if needed
        self.result_saver.save_decision_tracker(self.simulation_folder, self.company.decision_tracker)

        print(f"All simulation data saved to: {self.simulation_folder}")

        return results
# %% md

import concurrent.futures
import os
import time


def run_team_composition_simulation(team_name, team_config, random_seed, num_agents, years, result_dir):
    """Run a simulation for a specific team composition of MBTI types"""
    print(f"Starting simulation for team: {team_name} - {team_config['description']}")

    # Create and run simulation with specific team composition
    simulation = Simulation(
        agent_num=num_agents,
        random_seed=random_seed,
        model=model,
        result_dir=result_dir
    )

    # Update the initialization to use team_composition
    simulation.team_config = team_config
    simulation.profile = simulation.setup_profile(team_composition=team_config)

    # Re-initialize other components with the new profile
    simulation.BackgroundInfo = BackgroundInfo(num_agents - 1, simulation.profile['ceo']['profile']['name'])
    print('Setup background info')

    simulation.shareholders = simulation.setup_shareholder()
    print('Setup shareholders')

    simulation.ceo = CEO(simulation.profile['ceo'], simulation.BackgroundInfo, simulation.model)
    print('Setup CEO')

    # Create the simulation folder with original format - ceo_mbti is the identifier
    ceo_mbti = simulation.ceo.config.traits  # This stays the same
    simulation.simulation_folder = simulation.result_saver.create_simulation_folder(num_agents, ceo_mbti)

    # Save team info in a separate file within the folder
    os.makedirs(simulation.simulation_folder, exist_ok=True)
    team_info_path = os.path.join(simulation.simulation_folder, "team_info.txt")
    with open(team_info_path, "w") as f:
        f.write(f"Team Name: {team_name}\n")
        f.write(f"Description: {team_config['description']}\n")
        f.write(f"CEO MBTI: {team_config['ceo']}\n")
        f.write("Shareholder MBTI composition:\n")
        for mbti, count in team_config['shareholders'].items():
            f.write(f"- {mbti}: {count}\n")

    simulation.market = Market(seed=42)
    print('Setup market')

    simulation.company = Company(simulation.shareholders, simulation.ceo, simulation.market, simulation.prompts,
                                 model=simulation.model, simulation=simulation)
    print('Setup company')

    # Run the simulation
    results = simulation.run(years=years)

    print(
        f"Team {team_name} completed - Final assets: ${results['final_assets']:.2f}, Growth: {results['growth_percentage']:.2f}%")

    return team_name, results

def main():
    # Define team compositions for different scenarios
    team_compositions = {
        "strategic_complementary": {
            "ceo": "INTJ",
            "shareholders": {
                "ENTJ": 1,
                "ISTJ": 1,
                "ESTJ": 1,
                "ENFP": 1,
                "ISFJ": 1
            },
            "description": "Strategic Complementary Team"
        },
        "decision_quality_optimization": {
            "ceo": "ESTJ",
            "shareholders": {
                "ENTJ": 1,
                "INFP": 1,
                "ENFP": 1,
                "ISTP": 1,
                "ESFJ": 1
            },
            "description": "Decision Quality Optimization Team"
        },
        "innovation_enhancement": {
            "ceo": "ENFP",
            "shareholders": {
                "ENTP": 1,
                "ISTJ": 1,
                "ISFJ": 1,
                "INTJ": 1,
                "ESFP": 1
            },
            "description": "Innovation Enhancement Team"
        },
        "adaptive_decision_making": {
            "ceo": "ISTP",
            "shareholders": {
                "ENTJ": 1,
                "ESTJ": 1,
                "INFP": 1,
                "INTP": 1,
                "ESFJ": 1
            },
            "description": "Adaptive Decision Making Team"
        }
    }

    # Configuration
    random_seed = 42
    num_agents = 6  # CEO + 5 shareholders
    years = 1
    max_workers = 2  # Adjust based on your system's capabilities
    result_dir = "result/exp_diff_mbti"  # New directory for team composition results

    print(f"Running simulations for team compositions...")
    all_results = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_team = {}
        for team_name, team_config in team_compositions.items():
            future = executor.submit(
                run_team_composition_simulation,
                team_name,
                team_config,
                random_seed,
                num_agents,
                years,
                result_dir
            )
            future_to_team[future] = team_name

        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_team):
            team_name = future_to_team[future]
            try:
                completed_team, results = future.result()
                all_results[completed_team] = results
                print(f"Successfully completed simulation for {completed_team}")
            except Exception as exc:
                print(f"Team {team_name} generated an exception: {exc}")

    # Print summary of all results
    print("\n=== FINAL SUMMARY ===")
    print("Team | Final Assets | Growth Rate")
    print("-" * 45)
    for team_name, results in sorted(all_results.items()):
        desc = team_compositions[team_name]["description"]
        print(f"{desc:30} | ${results['final_assets']:11.2f} | {results['growth_percentage']:6.2f}%")

    return all_results


if __name__ == "__main__":

    start_time = time.time()
    all_results = main()
    end_time = time.time()
    duration = end_time - start_time
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = duration % 60

    print(f"\nAll simulations took {hours}h {minutes}m {seconds:.2f}s.")