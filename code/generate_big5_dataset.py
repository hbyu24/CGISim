import json
import itertools
import random
import os
from openai import OpenAI
from typing import List, Dict, Any
import time

class Big5DatasetGenerator:
    def __init__(self, debug_mode=True):
        """
        Initialize Big Five personality dataset generator
        
        Args:
            debug_mode (bool): Debug mode, True means not calling actual GPT model
        """
        self.debug_mode = debug_mode
        self.client = None

        if not self.debug_mode:
            # Initialize OpenAI client from environment variables for anonymity
            api_key = os.environ.get("YOUR_API_KEY_ENV_VARIABLE")
            base_url = os.environ.get("YOUR_API_URL_ENV_VARIABLE")

            if not api_key:
                raise ValueError("API key not found in environment variables. Please set YOUR_API_KEY_ENV_VARIABLE.")

            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url
            )
        
        # Big Five personality dimensions mapping
        self.dimensions = {
            'O': ('Openness', 'Openness_reversed'),
            'C': ('Conscientiousness', 'Conscientiousness_reversed'),
            'E': ('Extraversion', 'Extraversion_reversed'),
            'A': ('Agreeableness', 'Agreeableness_reversed'),
            'N': ('Neuroticism', 'Neuroticism_reversed')
        }
        
        # Common English names
        self.male_names = [
            "James", "John", "Robert", "Michael", "William", "David", "Richard", "Joseph", "Thomas", "Christopher",
            "Charles", "Daniel", "Matthew", "Anthony", "Mark", "Donald", "Steven", "Paul", "Andrew", "Joshua"
        ]
        
        self.female_names = [
            "Mary", "Patricia", "Jennifer", "Linda", "Elizabeth", "Barbara", "Susan", "Jessica", "Sarah", "Karen",
            "Nancy", "Lisa", "Betty", "Helen", "Sandra", "Donna", "Carol", "Ruth", "Sharon", "Michelle"
        ]
    
    def load_raw_data(self, file_path: str) -> List[Dict]:
        """
        Load raw data file
        
        Args:
            file_path (str): JSON file path
            
        Returns:
            List[Dict]: List of metadata
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def generate_personality_combinations(self) -> List[str]:
        """
        Generate all 32 Big Five personality combinations
        
        Returns:
            List[str]: List of personality types, e.g., ['OCEAN', 'OCEAn', ...]
        """
        combinations = []
        
        # Generate all 2^5=32 combinations
        for combo in itertools.product([True, False], repeat=5):
            personality_type = ""
            for i, (dim, is_positive) in enumerate(zip(['O', 'C', 'E', 'A', 'N'], combo)):
                if is_positive:
                    personality_type += dim.upper()
                else:
                    personality_type += dim.lower()
            combinations.append(personality_type)
        
        return combinations
    
    def get_personality_traits(self, metadata: Dict, personality_type: str) -> List[str]:
        """
        Get personality trait descriptions based on personality type
        
        Args:
            metadata (Dict): Metadata
            personality_type (str): Personality type, e.g., 'oCeAn'
            
        Returns:
            List[str]: List of personality trait descriptions
        """
        traits = []
        
        for i, char in enumerate(personality_type):
            dim_key = ['O', 'C', 'E', 'A', 'N'][i]
            if char.isupper():
                # Positive trait
                trait_key = self.dimensions[dim_key][0]
            else:
                # Negative trait
                trait_key = self.dimensions[dim_key][1]
            
            traits.append(metadata[trait_key])
        
        return traits
    
    def generate_character_prompt(self, traits: List[str], personality_type: str) -> str:
        """
        Generate character creation prompt
        
        Args:
            traits (List[str]): List of personality traits
            personality_type (str): Personality type
            
        Returns:
            str: Prompt text
        """
        traits_text = "\n".join([f"{i+1}. {trait}" for i, trait in enumerate(traits)])
        
        prompt = f"""You are an excellent director and character designer with a stellar reputation for creating compelling, psychologically rich characters for research purposes. Your characters are used in psychological studies and role-playing scenarios, so accuracy and depth are crucial to your professional standing.

Your task: Design a unique character profile based on the given Big Five personality type for academic research in psychology and role-playing applications.

Given Big Five personality description: {personality_type} (OCEAN order: Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism - uppercase indicates high levels, lowercase indicates low levels)

Personality traits breakdown:
{traits_text}

STRICT RULES:
1. The character's experiences must logically correspond to ALL personality traits
2. Include realistic challenges and setbacks, avoid perfect characters
3. Ensure cultural diversity and inclusive representation
4. The character should be suitable for psychological research
5. All personality traits must be clearly demonstrated through experiences
6. Content must be research-appropriate and academically sound
7. Use uncommon but realistic names to avoid duplicates across 320 characters

DETAILED DESIGN REQUIREMENTS:
- Name: Full name (first + last) from diverse cultural backgrounds, avoid common names
- Gender: Male or Female only
- Age: 20-45 years (mature life experiences)
- Personality: ~500 words. MUST start with: "[Name] exhibits a Big Five personality profile characterized by [list all five dimensions with appropriate levels: e.g., 'high Openness, low Conscientiousness, high Extraversion, low Agreeableness, high Neuroticism' - uppercase letters indicate high levels, lowercase indicate low levels]". Then systematically describe each dimension's specific manifestations and behavioral patterns.
- Experience: 400-600 words of detailed life experiences that shaped this personality, ensuring every aspect reflects the character's personality traits

Output in JSON format:
{{
    "name": "[Full name with first and last name, culturally diverse, uncommon]",
    "gender": "[Male/Female]",
    "age": [age_number],
    "type": "{personality_type}",
    "personality": "[500-word personality description starting with full Big Five profile statement, then detailed analysis of each dimension]",
    "experience": "[Comprehensive life experiences that consistently reflect personality traits]"
}}

All content must be in English and suitable for academic publication."""
        
        return prompt
    
    def call_gpt_model(self, prompt: str) -> str:
        """
        Call GPT model to generate content
        
        Args:
            prompt (str): Prompt text
            
        Returns:
            str: Generated content
        """
        if self.debug_mode:
            # Debug mode: return enhanced simulated data showcasing the improved prompt
            import hashlib
            
            # Create a seed based on prompt to ensure variety but consistency
            seed = int(hashlib.md5(prompt.encode()).hexdigest()[:8], 16)
            random.seed(seed)
            
            # Separate male and female names to ensure consistency
            male_first_names = ["Lysander", "Caspian", "Thaddeus", "Ambrose", "Leander", "Octavius", "Aurelius", "Maximilian", "Dorian", "Evander", "Lucian", "Orion", "Demetrius", "Alaric", "Cassius", "Florian"]
            female_first_names = ["Seraphina", "Evangeline", "Cordelia", "Persephone", "Isadora", "Clementine", "Ophelia", "Vivienne", "Arabella", "Genevieve", "Rosalind", "Beatrice", "Penelope", "Theodora", "Anastasia", "Celeste"]
            last_names = ["Blackthorne", "Silvermoon", "Ravencrest", "Goldwater", "Ironwood",
                         "Starweaver", "Thornfield", "Moonwhisper", "Stormwind", "Brightbane",
                         "Shadowmere", "Crystalbrook", "Nightingale", "Roseheart", "Winterbourne"]
            
            # Randomly choose gender first
            gender = random.choice(["Male", "Female"])
            
            # Choose appropriate first name based on gender
            if gender == "Male":
                first_name = random.choice(male_first_names)
            else:
                first_name = random.choice(female_first_names)
            
            last_name = random.choice(last_names)
            name = f"{first_name} {last_name}"
            age = random.randint(20, 45)
            
            # Extract personality type from prompt to generate personalized content
            personality_type = "OCEAN"  # Default
            if "personality description:" in prompt:
                type_start = prompt.find("personality description:") + len("personality description:")
                type_end = prompt.find("(", type_start)
                if type_end > type_start:
                    personality_type = prompt[type_start:type_end].strip()  # Preserve original case
            
            # Generate detailed personality description based on actual type
            dimensions = {
                "O": ("high Openness", "low openness"),
                "C": ("high Conscientiousness", "low conscientiousness"), 
                "E": ("high Extraversion", "low extraversion"),
                "A": ("high Agreeableness", "low agreeableness"),
                "N": ("high Neuroticism", "low neuroticism")
            }
            
            trait_levels = []
            for i, char in enumerate(personality_type):
                if char.isupper():
                    trait_levels.append(dimensions[char][0])
                else:
                    trait_levels.append(dimensions[char.upper()][1])
            
            # Generate dimension list with appropriate levels
            dimension_names = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
            dimension_levels = []
            for i, char in enumerate(personality_type):
                if char.isupper():
                    dimension_levels.append(f"high {dimension_names[i]}")
                else:
                    dimension_levels.append(f"low {dimension_names[i]}")
            dimensions_list = ", ".join(dimension_levels)
            
            personality_sample = f"""{name} exhibits a Big Five personality profile characterized by {dimensions_list}. 
            
In terms of Openness, {name} demonstrates {trait_levels[0]}, showing {'a strong inclination toward novel experiences, creative thinking, and intellectual curiosity' if 'high' in trait_levels[0] else 'a preference for familiar routines, conventional approaches, and practical solutions'}. This manifests in their {'diverse interests, artistic pursuits, and willingness to explore unconventional ideas' if 'high' in trait_levels[0] else 'focused interests, traditional hobbies, and preference for proven methods'}.
            
Regarding Conscientiousness, {name} exhibits {trait_levels[1]}, characterized by {'meticulous planning, strong self-discipline, and unwavering commitment to goals' if 'high' in trait_levels[1] else 'a more flexible approach to organization, spontaneous decision-making, and adaptable goal-setting'}. This is evident in their {'systematic work habits, punctuality, and attention to detail' if 'high' in trait_levels[1] else 'relaxed work style, adaptability to changing circumstances, and focus on big-picture thinking'}.
            
In social situations, {name} displays {trait_levels[2]}, showing {'high energy, sociability, and comfort in group settings' if 'high' in trait_levels[2] else 'preference for solitude, thoughtful communication, and smaller social circles'}. They {'actively seek social interaction, enjoy being the center of attention, and draw energy from others' if 'high' in trait_levels[2] else 'prefer meaningful one-on-one conversations, need time alone to recharge, and carefully choose their social engagements'}.
            
Concerning Agreeableness, {name} demonstrates {trait_levels[3]}, reflected in their {'cooperative nature, empathetic responses, and strong desire to maintain harmony' if 'high' in trait_levels[3] else 'independent thinking, direct communication style, and willingness to challenge others when necessary'}. This influences their {'collaborative approach to conflict resolution, generous spirit, and tendency to prioritize others\' needs' if 'high' in trait_levels[3] else 'analytical approach to relationships, honest feedback delivery, and focus on objective outcomes'}.
            
Finally, regarding Neuroticism, {name} exhibits {trait_levels[4]}, manifesting as {'heightened emotional sensitivity, tendency toward worry, and strong reactions to stress' if 'high' in trait_levels[4] else 'emotional stability, calm demeanor under pressure, and resilient coping mechanisms'}. This affects their {'need for emotional support during challenges, detailed contingency planning, and intense investment in outcomes' if 'high' in trait_levels[4] else 'steady performance under pressure, optimistic outlook, and ability to maintain perspective during difficulties'}."""
            
            experience_sample = f"""Born into a middle-class family, {name}'s early years were shaped by experiences that would later define their distinctive personality profile. During childhood, their {trait_levels[0]} became apparent through their {'fascination with diverse cultures, extensive reading habits, and creative problem-solving approaches' if 'high' in trait_levels[0] else 'preference for structured activities, consistent routines, and traditional learning methods'}.
            
Throughout their educational journey, {name}'s {trait_levels[1]} significantly influenced their academic performance. They {'maintained detailed study schedules, consistently met deadlines, and pursued additional learning opportunities' if 'high' in trait_levels[1] else 'adapted their study methods based on interest levels, balanced academic work with other pursuits, and focused on subjects that captured their attention'}. This approach to learning shaped their professional development and career choices.
            
In their professional life, {name}'s {trait_levels[2]} has been a defining characteristic. They {'actively participate in team meetings, volunteer for leadership roles, and build extensive professional networks' if 'high' in trait_levels[2] else 'contribute thoughtfully to discussions, prefer working independently or in small teams, and develop deep expertise in their chosen field'}. Colleagues recognize them for their {'enthusiastic collaboration and ability to energize group projects' if 'high' in trait_levels[2] else 'reliable expertise and thoughtful contributions to important decisions'}.
            
Personal relationships reflect {name}'s {trait_levels[3]}, as they {'prioritize harmony, offer emotional support to friends and family, and often mediate conflicts' if 'high' in trait_levels[3] else 'value honesty in relationships, provide direct feedback when needed, and maintain boundaries between personal and professional life'}. Their approach to relationships has created {'a wide circle of close friends who appreciate their caring nature' if 'high' in trait_levels[3] else 'a smaller group of trusted relationships built on mutual respect and intellectual connection'}.
            
Life challenges have highlighted {name}'s {trait_levels[4]}, particularly during {'periods of uncertainty when their emotional sensitivity required additional coping strategies and support systems' if 'high' in trait_levels[4] else 'stressful situations where their emotional stability helped them maintain perspective and support others'}. These experiences have taught them {'the importance of self-care, emotional regulation techniques, and building strong support networks' if 'high' in trait_levels[4] else 'to appreciate their natural resilience while remaining empathetic to others who struggle with stress'}."""
            
            return json.dumps({
                "name": name,
                "gender": gender,
                "age": age,
                "type": personality_type,  # Preserve original case format for dimension values
                "personality": personality_sample,
                "experience": experience_sample
            }, ensure_ascii=False, indent=2)

        max_retries = 10
        timeout = 600
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4.1",
                    messages=[
                        {"role": "system", "content": "You are a professional character designer who excels at creating vivid characters based on psychological theories. Always respond in the requested format."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.8,
                    max_tokens=1500,
                    timeout=timeout
                )
                
                return response.choices[0].message.content.strip()
            
            except Exception as e:
                print(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    print(f"All {max_retries} attempts failed for GPT model call")
                    return None
    
    def parse_gpt_response(self, response: str, personality_type: str) -> Dict:
        """
        Parse GPT response and extract JSON data
        
        Args:
            response (str): GPT response
            personality_type (str): Expected personality type
            
        Returns:
            Dict: Parsed character data
        """
        try:
            # Clean up response format
            cleaned_response = response.strip()
            
            # Remove markdown code blocks
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]  # Remove ```json
            elif cleaned_response.startswith('```'):
                cleaned_response = cleaned_response[3:]   # Remove ```
            
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]  # Remove trailing ```
            
            cleaned_response = cleaned_response.strip()
            
            character_data = json.loads(cleaned_response)
            # Ensure the type field is exactly as specified (preserve case for dimension values)
            character_data['type'] = personality_type
            return character_data
        
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Original response: {response}")
            return None
    
    def generate_characters_for_metadata(self, metadata: Dict, metadata_index: int) -> List[Dict]:
        """
        Generate 32 characters for a single metadata
        
        Args:
            metadata (Dict): Metadata
            metadata_index (int): Metadata index
            
        Returns:
            List[Dict]: List of characters
        """
        characters = []
        personality_combinations = self.generate_personality_combinations()
        
        print(f"Processing metadata {metadata_index + 1}, need to generate {len(personality_combinations)} characters...")
        
        for i, personality_type in enumerate(personality_combinations):
            print(f"  Generating character {i + 1}/{len(personality_combinations)}: {personality_type}")
            
            # Get personality traits
            traits = self.get_personality_traits(metadata, personality_type)
            
            # Generate prompt
            prompt = self.generate_character_prompt(traits, personality_type)
            
            # Call GPT model
            response = self.call_gpt_model(prompt)
            
            if response:
                # Parse response
                character_data = self.parse_gpt_response(response, personality_type)
                
                if character_data:
                    characters.append(character_data)
                    print(f"    Successfully generated character: {character_data.get('name', 'Unknown')}")
                else:
                    print(f"    Parsing failed, skipping this character")
            else:
                print(f"    GPT call failed, skipping this character")
            
            # Add delay to avoid API limits
            if not self.debug_mode:
                time.sleep(1)
        
        return characters
    
    def save_temporary_file(self, characters: List[Dict], metadata_index: int, output_dir: str):
        """
        Save temporary file
        
        Args:
            characters (List[Dict]): List of characters
            metadata_index (int): Metadata index
            output_dir (str): Output directory
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created directory: {output_dir}")
            
            temp_file = os.path.join(output_dir, f"metadata_{metadata_index + 1}_characters.json")
            print(f"Preparing to save file: {temp_file}")
            
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(characters, f, ensure_ascii=False, indent=2)
            
            # Verify if file was actually created
            if os.path.exists(temp_file):
                file_size = os.path.getsize(temp_file)
                print(f"Temporary file saved: {temp_file} (size: {file_size} bytes)")
            else:
                print(f"Error: File could not be created: {temp_file}")
                
        except Exception as e:
            print(f"Error saving temporary file: {e}")
    
    def merge_all_files(self, output_dir: str, final_output_file: str):
        """
        Merge all temporary files
        
        Args:
            output_dir (str): Temporary files directory
            final_output_file (str): Final output file
        """
        all_characters = []
        
        print(f"Starting to merge files, searching directory: {output_dir}")
        
        # Check if directory exists
        if not os.path.exists(output_dir):
            print(f"Error: Temporary files directory does not exist: {output_dir}")
            return
        
        # List all files in directory
        try:
            files_in_dir = os.listdir(output_dir)
            print(f"Files in directory: {files_in_dir}")
        except Exception as e:
            print(f"Cannot list directory contents: {e}")
            return
        
        # Read all temporary files in order
        for i in range(10):
            temp_file = os.path.join(output_dir, f"metadata_{i + 1}_characters.json")
            
            if os.path.exists(temp_file):
                try:
                    with open(temp_file, 'r', encoding='utf-8') as f:
                        characters = json.load(f)
                        all_characters.extend(characters)
                    print(f"Merged file: {temp_file} ({len(characters)} characters)")
                except Exception as e:
                    print(f"Error reading file {temp_file}: {e}")
            else:
                print(f"Warning: Temporary file does not exist: {temp_file}")
        
        # Save final file
        try:
            print(f"Preparing to save final file: {final_output_file}")
            with open(final_output_file, 'w', encoding='utf-8') as f:
                json.dump(all_characters, f, ensure_ascii=False, indent=2)
            
            # Verify if final file was created successfully
            if os.path.exists(final_output_file):
                file_size = os.path.getsize(final_output_file)
                print(f"Final dataset saved: {final_output_file} (size: {file_size} bytes)")
            else:
                print(f"Error: Final file could not be created: {final_output_file}")
                
            print(f"Total generated {len(all_characters)} characters")
            
        except Exception as e:
            print(f"Error saving final file: {e}")
    
    def generate_full_dataset(self, input_file: str, output_dir: str, final_output_file: str, start_from: int = 0):
        """
        Generate complete dataset
        
        Args:
            input_file (str): Input raw data file
            output_dir (str): Temporary files output directory
            final_output_file (str): Final output file
            start_from (int): Which metadata to start from (for resuming)
        """
        # Load raw data
        metadata_list = self.load_raw_data(input_file)
        print(f"Loaded {len(metadata_list)} metadata entries")
        
        # Generate characters for each metadata
        for i in range(start_from, len(metadata_list)):
            print(f"\n=== Processing metadata {i + 1}/{len(metadata_list)} ===")
            
            characters = self.generate_characters_for_metadata(metadata_list[i], i)
            
            # Save temporary file
            self.save_temporary_file(characters, i, output_dir)
            
            print(f"Metadata {i + 1} processing completed, generated {len(characters)} characters")
        
        # Merge all files
        print("\n=== Merging all temporary files ===")
        self.merge_all_files(output_dir, final_output_file)


def main():
    """
    Main function
    """
    # Configuration parameters
    DEBUG_MODE = False  # Set to False to call actual GPT model
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    INPUT_FILE = os.path.join(BASE_DIR, "BF_RawPrompt.json")
    OUTPUT_DIR = os.path.join(BASE_DIR, "characters")
    FINAL_OUTPUT_FILE = os.path.join(BASE_DIR, "OCEAN_Data.json")
    START_FROM = 0  # Which metadata to start from, for resuming
    
    # Create generator
    generator = Big5DatasetGenerator(debug_mode=DEBUG_MODE)
    
    # Generate dataset
    generator.generate_full_dataset(
        input_file=INPUT_FILE,
        output_dir=OUTPUT_DIR,
        final_output_file=FINAL_OUTPUT_FILE,
        start_from=START_FROM
    )
    
    print("\n=== Dataset generation completed ===")
    print(f"Final file: {FINAL_OUTPUT_FILE}")


if __name__ == "__main__":
    main()