import gradio as gr
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage
from agent import simple_agent

class SimpleAgent:
    """A langgraph agent."""
    def __init__(self):
        print("SimpleAgent initialized.")
        self.graph = simple_agent

    def __call__(self, question: str) -> str:
        # print(f"Agent received question (first 50 chars): {question[:50]}...")
        # Wrap the question in a HumanMessage from langchain_core
        messages = [HumanMessage(content=question)]
        messages = self.graph.invoke({"messages": messages})
        # answer = messages['messages'][-1].content
        # m = re.search(r"FINAL ANSWER:\s*(.*)", answer, flags=re.DOTALL)
        # if m:
        #     result = m.group(1)
        # else:
        #     result = ""
        answer = messages["messages"][-1].content
        return answer
    
agent = SimpleAgent()

with gr.Blocks() as demo:
    gr.Markdown("# I am a SImple AI Agent\n")
    gr.Markdown("""
                I will try my best to respond to your need. Besides chatting, abilities include:
                1. Surfing web to gather information
                2. Surfing WIkipedia to extract information
                3. Listen to audio and transcribe
                4. Watch YouTube video and count maximum number of a certain object
                5. Listen to YouTube video and transcribe, I can also answer question from there.
                6. Perform basic mathematical operation.
                7. Extract information from specified ArXiv papers.
                8. Run a given python code (experimental).
                """)
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Type here…")
    send = gr.Button("Send")
    def respond(user_msg, history):
        agent_reply = agent(user_msg)
        history = history + [(user_msg, agent_reply)]
        return history, ""
    send.click(respond, inputs=[msg, chatbot], outputs=[chatbot, msg])
demo.launch()