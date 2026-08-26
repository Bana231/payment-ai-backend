from agent_graph import investigation_graph

result = investigation_graph.invoke(
    {
        "question": "Why are payment transactions failing?"
    }
)

print(result)