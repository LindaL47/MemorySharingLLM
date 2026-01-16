import os
import torch
import Agent_MS
import Retrival_model
import Evaluate
import json
from tqdm import tqdm

# 强制指定使用GPU 2（对应物理机空闲卡），统一所有模块的设备
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
torch.cuda.empty_cache()  # 清空GPU缓存
# 定义全局设备，让Retrival_model/Evaluate/Agent_MS都复用这个配置
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")
# 将设备传递给其他模块（需确保其他模块支持接收）
Retrival_model.DEVICE = DEVICE
Evaluate.DEVICE = DEVICE
Agent_MS.DEVICE = DEVICE

if __name__ == '__main__':
    print("yunxing")
    #Retrival_model.main("./LiteralPool.jsonl")
    print("jiazai")

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
    with open("./Literalproblem.jsonl", 'r', encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)  # Load JSON object from each line
            problem_list.append(data["messages"][0]["content"])
    count = 0
    total_count = 0
    for problem in tqdm(problem_list):
        all_sentences = Evaluate.allSentences("./LiteralPool.jsonl")
        final_question = Evaluate.final_prompt(problem, "model.pth", "optimizer.pth", all_sentences)
        answer = Evaluate.chatgpt_answer(final_question)
        total_count = total_count + 1
        # Set training=False to skip the time-consuming fine-tuning process during testing
        bValue = Agent_MS.message_store(final_question, answer, "./LiteralPool.jsonl", "Literature", training=False)
        if bValue:
            count = count + 1
        if count % 20 == 0:
            print("total_Count:" + str(total_count))
            print("Count:" + str(count))
            print("For Limerick: ----------------------------")
            Evaluate.main("LiteralPool.jsonl", "limerickTest.jsonl")
            print("For wuyanlvshi:--------------------------")
            Evaluate.main("LiteralPool.jsonl", "lvshiTest.jsonl")
            print("For sonnet")
            Evaluate.main("LiteralPool.jsonl", "sonnetTest.jsonl")

