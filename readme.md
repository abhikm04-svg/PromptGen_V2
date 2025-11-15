<https://promptgen-gemini.streamlit.app/#prompt-optimizer>
# 🧙 PromptGen V2: AI-Powered Prompt Optimizer

PromptGen V2 is a sophisticated, multi-agent application designed to help you craft the perfect prompt for your use case. Using a collaborative workflow, 
this tool takes your initial idea and iteratively refines it into a high-quality, optimized prompt ready for use with powerful generative AI models like Google Gemini.

The application is built with Streamlit, providing an intuitive and interactive user interface to guide you through the prompt engineering process.

## ✨ Features

- **Interactive Workflow**: Instead of guessing, the app asks you targeted questions to understand your exact needs.
- **Multi-Agent System**: Utilizes specialized AI agents for different tasks:
    - **Question Generator**: Asks clarifying questions to flesh out your idea.
    - **Prompt Generator**: Creates a well-structured, XML-tagged prompt.
    - **Prompt Tester**: Executes the prompt to see what kind of output it produces.
    - **Prompt Analyzer**: Scores the prompt and its output against a strict set of benchmarks, providing actionable feedback.
- **Automated Optimization Loop**: The system automatically iterates on your prompt based on the analyzer's feedback, striving for a perfect score.
- **Detailed Results**: View the final score, the number of iterations it took, and whether the prompt reached the target score.
- **Iteration History**: Expand and review each step of the optimization process to see how the prompt evolved.
- **Downloadable Assets**:
    - Download the final optimized prompt as an `.xml` file.
    - Download a graph visualizing the token usage throughout the optimization process.

## 🚀 How It Works

The application follows a structured workflow to ensure the best possible outcome:

1.  **Idea Submission**: You start by providing a basic idea for a prompt.
2.  **Clarification**: The Question Generator agent asks 3-5 questions to gather details about your goal, audience, desired tone, and output format.
3.  **Initial Generation**: Once you answer the questions, the Prompt Generator agent creates the first version of your prompt.
4.  **Test & Analyze**: The Prompt Tester executes the prompt, and the Prompt Analyzer scores the result.
5.  **Iterative Refinement**: If the score is below 100, the analyzer's feedback is fed back to the Prompt Generator to create an improved version.
    This loop continues for a set number of iterations or until the target score is achieved.
7.  **Final Prompt**: The best prompt from the optimization loop is presented to you.

## 🛠️ Tech Stack

- **Backend**: Python
- **AI Models**: Google Gemini (`gemini-2.5-flash` and `gemini-2.5-pro`)
- **Framework**: Streamlit
- **Core Libraries**: `google-generativeai`, `pandas`, `matplotlib`

## ⚙️ Setup and Installation

To run this application locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone <https://github.com/abhikm04-svg/PromptGen_V2>
    cd PromptGen_V2
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your API Key:**
    The application requires a Google Gemini API key. You need to set it up as a Streamlit secret.
    - Create a file named `.streamlit/secrets.toml` in the root of the project directory.
    - Add your API key to this file as follows:
      ```toml
      GOOGLE_API_KEY = "YOUR_API_KEY_HERE"
      ```

5.  **Run the Streamlit application:**
    ```bash
    streamlit run promptgen_app.py
    ```

The application should now be running in your web browser!

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements or new features, 
please feel free to open an issue or submit a pull request.
