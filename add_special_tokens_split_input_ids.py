def add_special_tokens_split_input_ids(split_input_ids, tokenizer):
    '''
    adds the start [CLS], end [SEP] and padding [PAD] special tokens to the list of split_input_ids
    '''
    
    # add start and end
    split_input_ids_with_tokens = [[tokenizer.cls_token_id]+s+[tokenizer.sep_token_id] for s in split_input_ids]
    
    # add padding to the last one
    split_input_ids_with_tokens[-1] = split_input_ids_with_tokens[-1]+[tokenizer.pad_token_id 
                                                                       for _ in range(len(split_input_ids_with_tokens[0])-len(split_input_ids_with_tokens[-1]))]
    
    return(split_input_ids_with_tokens)
    