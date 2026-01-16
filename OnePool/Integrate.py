import Agent_MS
import Retrival_model
import Evaluate
import json
from tqdm import tqdm

if __name__ == '__main__':
    Retrival_model.main("./SinglePool.jsonl")

    """
    while True:
        all_sentences = Evaluate.allSentences("./wuyanlvshi.jsonl")
        user_input = input("Question:")
        agent_type = input("Select one from: Literature, Logic, Translate")
        final_question = Evaluate.final_prompt(user_input, "model.pth", "optimizer.pth", all_sentences)
        answer = Evaluate.chatgpt_answer(final_question)
        Agent_MS.message_store(final_question, answer, "./train.jsonl", agent_type)
        print(final_question)
        print(answer)
        if user_input.lower() == 'exit':
            print("Exiting the program.")
            break  # This exits the loop #------------------------------------
    """
    problem_list = []
    with open("./TotalProblem.jsonl", 'r', encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)  # Load JSON object from each line
            problem_list.append(data["messages"][0]["content"])
    count = 0
    total_count = 0
    for problem in tqdm(problem_list):
        all_sentences = Evaluate.allSentences("./SinglePool.jsonl")
        final_question = Evaluate.final_prompt(problem, "model.pth", "optimizer.pth", all_sentences)
        answer = Evaluate.chatgpt_answer(final_question)
        totot_count = total_count + 1
        bValue = Agent_MS.message_store(final_question, answer, "./SinglePool.jsonl", "Total")
        if bValue:
            count = count + 1
        if count % 20 == 0:
            print("total_Count:" + str(total_count))
            print("Count:" + str(count))
    print("For wuyanlvshi: ----------------------------")
    Evaluate.main("./SinglePool.jsonl", "./lvshiTest.jsonl")
    print("For Limerick:--------------------------")
    Evaluate.main("./SinglePool.jsonl", "./limerickTest.jsonl")
    print("For sonnet")
    Evaluate.main("./SinglePool.jsonl", "./sonnetTest.jsonl")
    print("For puns: ----------------------------")
    Evaluate.main("./SinglePool.jsonl", "./punsTest.jsonl")
    print("For puzzle:--------------------------")
    Evaluate.main("./SinglePool.jsonl", "./puzzleTest.jsonl")
    print("For riddles")
    Evaluate.main("./SinglePool.jsonl", "./riddlesTest.jsonl")
    print("For fitness: ----------------------------")
    Evaluate.main("./SinglePool.jsonl", "./fitnessTest.jsonl")
    print("For study:--------------------------")
    Evaluate.main("./SinglePool.jsonl", "./studyTest.jsonl")
    print("For travel")
    Evaluate.main("./SinglePool.jsonl", "./travelTest.jsonl")

