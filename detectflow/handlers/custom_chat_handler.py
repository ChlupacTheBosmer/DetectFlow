import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM
import bitsandbytes, flash_attn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Dict, Any


class CustomChatHandler:
    """
    AIAssistant class for generating natural language responses using a pretrained Llama model
    optimized for running on CUDA-enabled devices using 4-bit quantization.

    Attributes:
        device (torch.device): Device on which the model will be loaded (CUDA).
        tokenizer (transformers.AutoTokenizer): Tokenizer for converting text to tokens.
        model (transformers.LlamaForCausalLM): Pretrained Llama causal language model.
    """

    def __init__(self, model_name='NousResearch/Hermes-2-Pro-Llama-3-8B'):
        """
        Initializes the AIAssistant instance by setting up the tokenizer and model.

        Args:
            model_name (str): Hugging Face model identifier. Defalt model info: https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B

        Raises:
            EnvironmentError: If CUDA is not available on the system.
        """
        if not torch.cuda.is_available():
            raise EnvironmentError("CUDA is not available. AIAssistant requires a GPU to function efficiently.")

        self.device = torch.device("cuda")

        # Special tokens
        self._s = "<|im_start|>"
        self._e = "<|im_end|>"
        self._sys = "system"
        self._usr = "user"
        self._ass = ":assistant:"

        # Initialize tokenizer with trust_remote_code to safely load remote configuration.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # Configure quantization settings for the model.
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

        # Load the model with specified quantization configurations.
        self.model = LlamaForCausalLM.from_pretrained(model_name, quantization_config=quantization_config)
        # self.model.to(self.device)  # Move the model to the CUDA device.

    def extract_generated_text(self, prompt, full_response):
        # Define the marker that appears at the end of the prompt and before the response
        # You may need to adjust the exact string if it varies
        marker = ":assistant:"

        # Find the position of the marker in the full response
        marker_index = full_response.find(marker)

        if marker_index != -1:
            # Calculate start index of new text by adding the length of marker
            start_index = marker_index + len(marker)

            # Extract the new text from the response
            generated_text = full_response[start_index:]

            return generated_text.strip()

        # If the marker is not found, return an empty string or handle appropriately
        return ""

    def generate_response(self, prompt: str, max_tokens: int = 500, temperature: float = 0.5,
                          repetition_penalty: float = 1.1):
        """
        Generates a response based on the input prompt using the model.

        Args:
            prompt (str): The prompt text to feed into the model.
            max_tokens (int): The maximum number of new tokens to generate.
            temperature (float): Controls randomness in response generation.
            repetition_penalty (float): Increases / decreases likelihood of repetition.

        Returns:
            str: The generated natural language response.
        """
        try:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
            generated_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                eos_token_id=self.tokenizer.eos_token_id
            )
            response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=False,
                                             clean_up_tokenization_space=False)

            # Extract only the new generated text
            text = self.extract_generated_text(prompt, response)
            # text = response

            return text
        except Exception as e:
            print(f"Failed to generate response: {e}")
            return None

    @staticmethod
    def format_data_for_prompt(data):
        """
        Format a dictionary into a structured string for use in a LLM prompt.
        Handles various data types including str, int, dict, list, and tuple.

        Args:
        data (dict): A dictionary where keys are descriptive headers and values can be various data types.

        Returns:
        str: A formatted string that combines all the headers and their associated data.
        """
        formatted_string = ""
        for header, value in data.items():
            formatted_string += f"{header}:\n"  # Append the header

            # Check the type of value and format accordingly
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    formatted_string += f"  {subkey}: {subvalue}\n"
            elif isinstance(value, (list, tuple)):
                list_items = ", ".join(map(str, value))
                formatted_string += f"  {list_items}\n"
            else:  # For str, int, float, etc.
                formatted_string += f"  {value}\n"

            # Add a newline for separation between sections
            formatted_string += "\n"

        return formatted_string.strip()  # Remove the last extra newline

    def compose_job_status_notification(self, data: Dict, **kwargs: Dict[str, Any]):
        """
        Constructs a custom prompt and generates a response specific to PBS job failures.

        Args:
            data (dict): Dictionary containing data to be included in the prompt. Make sure the keys are informative descriptions of the values.

        Returns:
            str: Customized email informing the user of the job failure.
        """

        additional_data_str = CustomChatHandler.format_data_for_prompt(data)

        system_prompt = "You are a Python developer with extensive experience in data processing and the management of Portable Batch System (PBS) job scheduling. Your primary role is to analyze the status of PBS jobs, including outputs and error logs, and provide expert consultation to users on understanding and resolving job issues."

        user_prompt = f"Analyze the status of PBS job and if available output and error logs as well as the script running as part of the job, assess the situation, and write an email to the user. The email should summarize the situation, explain what was the error if any, whether job results were generated, and suggest solutions. Also, summarize actions needed from the user to fix the problem and rerun the job. The email should be clear, concise, comprehensive, and perfectly structured. Do not include the subject name nor any signature. Only refer to data and information passed in the prompt such as the error log or job info. Job data to inform your response: {additional_data_str}."

        prompt = f"""{self._s}{self._sys} {system_prompt}{self._e}{self._s}{self._usr} {user_prompt}{self._e} {self._s}{self._ass}"""

        print(prompt)
        return self.generate_response(prompt, **kwargs)