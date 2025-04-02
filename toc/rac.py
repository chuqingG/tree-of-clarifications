import dsp

def retrieve_passages(topk, question):
    # question = ins.question
    passages = dsp.retrieve(question, k=topk)
    
    return passages