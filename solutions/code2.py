def LSTMcell(prev_ht, prev_ct,input):
    combine = prev_ht + input
    forget_t = forget_layer(combine)
    candidate = candidate_layer(combine)
    input_t = input_layer(combine)
    c_t = candidate*input_t + prev_ct*input_t
    output_t = output_layer(combine)
    h_t = output_t*tanh(c_t)
    return h_t, c_t