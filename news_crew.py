import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool

# Load environment variables from .env file
load_dotenv()

# Initialize the Serper Search Tool
search_tool = SerperDevTool()

# 1. Researcher Agent (Has web search capabilities)
researcher = Agent(
    role="Tech News Researcher",
    goal="Find today's top 3 most important tech news stories efficiently.",
    backstory=(
        "You are a seasoned tech analyst who stays ahead of industry trends. "
        "You excel at searching the web for current, accurate, and high-impact technology news."
    ),
    tools=[search_tool],  # Only Researcher gets tools
    max_iter=3,            # Prevents unnecessary tool retry loops
    verbose=True
)

# 2. Writer Agent (No search tools, purely synthesizes research)
writer = Agent(
    role="Tech Newsletter Writer",
    goal="Synthesize researched tech news into a strict 5-line newsletter format.",
    backstory=(
        "You are a fastidious editor and tech writer. You specialize in taking complex news "
        "and summarizing it briefly and punchily for busy professionals."
    ),
    tools=[],  # Writer has NO tools to force reliance on Researcher's output
    verbose=True
)

# Task 1: Research Today's Tech News
find_news_task = Task(
    description=(
        "Search the web for today's top 3 tech news stories. "
        "Gather a summary and key points for each story."
    ),
    expected_output=(
        "A structured overview of the top 3 tech news stories of today, "
        "including headlines and brief summaries for each."
    ),
    agent=researcher
)

# Task 2: Format Newsletter (Uses output from Task 1)
write_newsletter_task = Task(
    description=(
        "Take the 3 tech stories provided from the research task and format them into a newsletter.\n"
        "Your final answer MUST contain strictly EXACTLY 5 lines (no extra headers, intro, or blank lines):\n"
        "Line 1: Tech in 5 lines\n"
        "Line 2: [Story 1 summary]\n"
        "Line 3: [Story 2 summary]\n"
        "Line 4: [Story 3 summary]\n"
        "Line 5: Why it matters: [Key takeaway sentence]"
    ),
    expected_output="An exact 5-line newsletter formatted strictly according to instructions.",
    agent=writer,
    context=[find_news_task]  # Hand over researcher's output to the writer
)

# 3. Define Sequential Crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[find_news_task, write_newsletter_task],
    process=Process.sequential,  # Tasks execute in order
    verbose=True
)

if __name__ == "__main__":
    result = crew.kickoff()
    print("\n================ FINAL NEWSLETTER ================\n")
    print(result)