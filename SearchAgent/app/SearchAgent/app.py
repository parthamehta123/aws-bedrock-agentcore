from agent import agent as search_agent
from bedrock_agentcore import BedrockAgentCoreApp

app = BedrockAgentCoreApp()

@app.entrypoint
def main_function(event: dict, context: dict):
    prompt = event.get("prompt", "")

    res = search_agent.invoke(
        {"messages": [ {"role": "user", "content": prompt} ]}
    )

    answer = res["messages"][-1].content

    return {
        "answer": answer
    }

if __name__ == "__main__":
    app.run(main_function)