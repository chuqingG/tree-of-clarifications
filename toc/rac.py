import dsp

def retrieve_passages(args, ins, bing_passages=None):
    question = ins.question
    passages = dsp.retrieve(question, k=args.top_k_docs)
    
    return passages