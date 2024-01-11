# split tokenized text into chunks for the model
def input_ids_splitter(input_ids, window_size=510, window_stride=255):
    '''
    Given a list of input_ids (tokenized text ready for a model),
    returns a list with chuncks of window_size, starting and ending with the special tokens (potentially with padding)
    the chuncks will have overlap by window_size-window_stride
    '''
        
    # int() rounds towards zero, so down for positive values
    num_splits = max(1, int(len(input_ids)/window_stride))
    
    split_input_ids = [input_ids[i*window_stride:i*window_stride+window_size] for i in range(num_splits)]
    
    
    return(split_input_ids)