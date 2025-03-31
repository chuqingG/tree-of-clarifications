import dsp

def retrieve_passages(args, ins):
    question = ins.question
    passages = dsp.retrieve(question, k=args.top_k_docs)
    
    return passages