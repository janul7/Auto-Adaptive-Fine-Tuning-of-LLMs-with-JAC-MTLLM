# Expert_Answer.jac

import from mtllm { Model }

glob llm = Model(model_name="gpt-4o-mini", verbose=True);

def get_expert(question: str) -> str by llm(method='Reason');
def get_answer(question: str, expert: str) -> str by llm();

with entry {
    while True {
        print("Ask a question (or type 'exit' to quit):");
        question = input();

        if question == "exit" {
            print("Exiting Expert Answer... 👋");
            break;
        }

        expert = get_expert(question);
        answer = get_answer(question, expert);

        print(f"{expert} says: '{answer}'\n");
    }
}
