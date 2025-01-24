# example_usage.py

from src.llm_interface import LLMInterface
from src.tools import WikipediaTool
from src.chain_of_thought import ChainOfThought
from src.react import ReAct
from src.langchain_agent import LangchainReActAgent
from src.self_consistency import SelfConsistency

def main():
    # Configuration
    PROJECT_ID = "YOUR_PROJECT_ID_HERE"
    LOCATION = "us-central1"
    MODEL_NAME = "text-bison@001"

    # Initialize interfaces and tools
    llm = LLMInterface(project_id=PROJECT_ID, location=LOCATION, model_name=MODEL_NAME)
    wiki_tool = WikipediaTool(return_chars=1000)

    # Initialize Chain of Thought
    chain_of_thought = ChainOfThought(llm=llm, tool=wiki_tool)

    # Example: Chain of Thought
    one_shot_exemplar = """Q: Roger has 5 tennis balls.
He buys 2 more cans of tennis balls.
Each can has 3 tennis balls. How many tennis balls does he have now?
A: Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11. The answer is 11.
Q: """
    question = """Nomfundo writes legal briefs.
Each brief has 3 sections, each section takes 4 hours.
She wrote 3 briefs this week. How long did it take?"""
    response = chain_of_thought.generate_response(exemplar=one_shot_exemplar, question=question)
    print("Chain of Thought Response:")
    print(response)

    # Initialize ReAct
    react = ReAct(llm=llm, tool=wiki_tool)

    # Example: ReAct
    context = """Answer questions with thoughts, actions, and observations.

Think about the next action to take. Then take an action.
All actions are a lookup of wikipedia.
The wikipedia action returns the beginning of the best-matching article.
When making a wikipedia lookup action, end the lookup with <STOP>.
After the wikipedia action, you will make an observation.
The observation is based on what you learn from the wikipedia lookup action.
After the observation, begin the loop again with a thought.

Repeat as necessary a thought, taking an action, and making an observation.
Keep repeating as necessary until you know the answer to the question.
When you think you have an answer, return the answer in the format:
"Answer[answer goes here between square brackets]"
as part of a thought. Make sure to capitalize "Answer".

Only use information in the observations to answer the question."""
    exemplar_react = """Example:
Question: Who was born first, Ronald Reagan or Gerald Ford?
Thought 1: I need to look up Ronald Reagan and see when he was born.
Action 1: Ronald Reagan<STOP>
Observation 1: Ronald Wilson Reagan (February 6, 1911 – June 5, 2004) was an American politician and actor who served as the 40th president of the United States from 1981 to 1989. A conservative, he was the first president from the West Coast and the first divorced president. Reagan was born in Tampico, Illinois, and raised in Dixon, Illinois. He was educated at Eureka College, where he studied economics and sociology. After graduating, Reagan moved to California, where he became a radio sports announcer. He later moved into acting, appearing in over 50 films. Reagan served as president of the Screen Actors Guild from 1947 to 1952.
Thought 2: Ronald Reagan was born in 1911. I need to look up Gerald Ford and see when he was born.
Action 2: Gerald Ford<STOP>
Observation 2: Gerald Rudolph Ford Jr. ( JERR-əld; born Leslie Lynch King Jr.; July 14, 1913 – December 26, 2006) was an American politician who served as the 38th president of the United States from 1974 to 1977. He previously served as the leader of the Republican Party in the U.S. House of Representatives from 1965 to 1973, when he was appointed the 40th vice president by President Richard Nixon, after Spiro Agnew's resignation. Ford succeeded to the presidency when Nixon resigned in 1974, but was defeated for election to a full term in 1976. Ford is the only person to become U.S. president without winning an election for president or vice president.
Ford was born in Omaha, Nebraska and raised in Grand Rapids, Michigan. He attended the University of Michigan, where he played for the school's football team before eventually attending Yale Law School. Afterward, he served in the U.S. Naval Reserve from 1942 to 1946. Ford began his political career in 1949 as the U.S. representative from Michigan's 5
Thought 3: Gerald Ford was born in 1913. 1911 is before 1913. Answer[Ronald Reagan]"""

    react_question = "When was the opening year of the theater that debuted Ibsen's 'A Doll's House'?"
    react_answer = react.react_chain(context=context, exemplar=exemplar_react, question=react_question, show_activity=True)
    print("ReAct Answer:")
    print(react_answer)

    # Initialize Langchain ReAct Agent
    langchain_agent = LangchainReActAgent(model_name=MODEL_NAME, project_id=PROJECT_ID, location=LOCATION)
    langchain_response = langchain_agent.run_query("What US President costarred with a chimp in 'Bedtime for Bonzo'?")
    print("Langchain ReAct Agent Response:")
    print(langchain_response)

    # Initialize Self-Consistency
    self_consistency = SelfConsistency(llm=llm, tool=wiki_tool)
    sc_prompt = """Factories have a baseline productivity of 100 units per day.
Not all factories have the baseline productivity.
When a factory is being upgraded, it has 25% of the baseline productivity.
When a factory is undergoing maintenance, it has 50% of the baseline.
When a factory is under labor action, it produces nothing.
Megacorp has 19 factories in total.
3 factories are being upgraded.
2 factories are under maintenance.
1 is under labor action.
How many units does megacorp produce in a day?"""
    sc_exemplar = """Q: A regular tennis ball can holds 5 balls.
A large tennis ball can holds 200% of a regular tennis ball can.
A small tennis ball can holds 40% of a regular tennis ball can.
A collectable tennis ball can holds no tennis balls.
Roger has 10 tennis ball cans.
3 cans are large cans.
4 cans are small cans.
1 can is collectable.
How many tennis balls does Roger have?
A: We need to find the number of regular tennis ball cans.
Roger has 10 (total) - 3 (large) - 4 (small) - 1 (collectable) = 2 regular cans.
A large tennis ball can holds 200% of 5 = 10 tennis balls.
A small tennis ball can holds 40% of 5 = 2 tennis balls.
Next count how many balls come from each can type.
3 large cans is 3 * 10 = 30 tennis balls.
4 small cans is 2 * 4 = 8 tennis balls.
2 regular cans is 2 * 5 = 10 tennis balls
1 collectable can is 0 tennis balls.
To get the answer, add the number of balls from each can type.
Roger has 30 (large) + 8 (small) + 10 (regular) + 0 (collectable) = 48 balls.
The answer is 48.
Q: """

    sc_answer_counts = self_consistency.run_multiple_responses(prompt=sc_prompt, parameters=llm.parameters, runs=40)
    self_consistency.plot_answer_distribution(sc_answer_counts)

if __name__ == "__main__":
    main()

