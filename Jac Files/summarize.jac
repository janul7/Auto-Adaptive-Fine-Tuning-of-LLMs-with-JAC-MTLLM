import from mtllm { Model }

glob llm = Model(model_name="gpt-4o-mini");

obj Summary {
  has summary: str;
}

def summarize(paragraph: str) -> Summary by llm();

with entry {
  while True {
    print("Paste a paragraph to summarize (or type 'exit' to quit):");
    user_text = input();

    if user_text == "exit" {
      print("Exiting summarizer... 👋");
      break;
    }

    result = summarize(user_text);
    print("--- Summary ---");
    print(result);
    print("");  # adds a blank line for readability
  }
}
