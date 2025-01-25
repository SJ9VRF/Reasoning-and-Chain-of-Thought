# Reasoning-and-Chain-of-Thought

![Screenshot_2025-01-24_at_4 13 04_PM-removebg-preview](https://github.com/user-attachments/assets/e09281eb-741d-4fa6-a331-a6a587d49fe9)

## Overview

**Advanced Prompting: Chain of Thought and ReAct** is a comprehensive Python repository designed to explore and implement advanced prompting strategies for Large Language Models (LLMs). Leveraging **Chain of Thought** and **ReAct (Reasoning + Acting)** paradigms, this project integrates with external tools like Wikipedia and utilizes platforms such as Google Cloud Vertex AI and Langchain to enhance the reasoning and action capabilities of LLMs.

## Features

- **Chain of Thought Prompting**: Enhances LLM reasoning by providing step-by-step exemplars.
- **ReAct Prompting**: Combines reasoning with actions to interact with external tools.
- **Langchain Integration**: Utilizes Langchain for building ReAct agents with built-in tool integrations.
- **Self-Consistency**: Implements techniques to improve the reliability and consistency of LLM responses.
- **Comprehensive Testing**: Includes unit tests for all modules using Python's `unittest` framework with mocked external dependencies.
- **Modular Design**: Organized into modular Python classes for scalability and maintainability.

## Installation

### Prerequisites

- **Google Cloud Account**: Required for accessing Vertex AI.
- **Google Cloud SDK**: Install and initialize the [Google Cloud SDK](https://cloud.google.com/sdk/docs/install).

### Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/advanced_prompting.git
   cd advanced_prompting
   ```
   
2. **Create a Virtual Environment (Optional but Recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

   
3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
