from torch.utils.data import Dataset
import numpy as np
import constants
import json
from vocab_util import word2charix

# Return (context, question) pairs sorted by length of context in terms of tokens
# (i.e., first return longer contexts)
class SquadDataset(Dataset):

    def __init__(self, file_ids_ctx, file_ids_q, file_ctx, file_q, file_span, char2ix_file):

        #This will be used in the character mapping
        with open(char2ix_file) as json_data:
            self.char2ix = json.load(json_data)

        list_context = []
        list_questions = []
        self.max_question_len = 0
        len_contexts = []

        list_context_raw = []
        list_question_raw = []
        list_span = []

        with open(file_ids_q, "r") as question_file_ids, \
                open(file_ids_ctx, "r") as context_file_ids, \
                open(file_q, "r") as question_file, \
                open(file_ctx, "r") as context_file, \
                open(file_span, "r") as span_file:

            # We need to find the max token length for questions
            # Also, collect the contexts and questions to sort them acc. to context length
            # (length in terms of tokens, NOT in terms of characters)
            for context, question, context_raw, question_raw, spans in \
                    zip(context_file_ids, question_file_ids, context_file, question_file, span_file):

                context_array = np.fromstring(context.strip('\n'), dtype=int, sep=' ')
                question_array = np.fromstring(question.strip('\n'), dtype=int, sep=' ')
                span_array = np.fromstring(spans.strip('\n'), dtype=int, sep=' ')

                list_context_raw.append(context_raw.strip('\n'))
                list_question_raw.append(question_raw.strip('\n'))

                list_context.append(context_array)
                list_questions.append(question_array)

                list_span.append(span_array)

                if question_array.shape[0] > self.max_question_len:
                    self.max_question_len = question_array.shape[0]

                len_contexts.append(context_array.shape[0])

        pos_for_batch = np.argsort(len_contexts)

        self.context_sorted = []
        self.question_sorted = []
        self.context_raw_sorted = []
        self.question_raw_sorted = []
        self.spans_sorted = []

        for i in pos_for_batch:
            self.context_sorted.append(list_context[i])
            self.question_sorted.append(list_questions[i])
            self.context_raw_sorted.append(list_context_raw[i])
            self.question_raw_sorted.append(list_question_raw[i])
            self.spans_sorted.append(list_span[i])

        del list_context
        del list_questions
        del list_span
        del len_contexts
        del list_context_raw
        del list_question_raw
        del pos_for_batch

    def __len__(self):
        return len(self.context_sorted)

    #Return:
    # 400 dimensional vector containing the indices of context tokens in the word embedding space
    # 60 dimensional vector containing the indices of question tokens in the word embedding space
    # 400x16 matrix containing the indices of context tokens in the char embedding space (16d for each word)
    # 60x16 matrix containing the indices of question tokens in the char embedding space (16d for each word)
    # 2 dimensional vector containing the start and end token position of the answer in the context
    # Raw context content (for debugging)
    # Raw question content (for debugging)
    #
    def __getitem__(self, idx):

        context_pad_word = np.array([constants.PAD_ID] * (constants.MAX_CONTEXT - self.context_sorted[idx].shape[0]))
        question_pad_word = np.array([constants.PAD_ID] * (self.max_question_len - self.question_sorted[idx].shape[0]))

        context_word_idx = np.concatenate((self.context_sorted[idx], context_pad_word))
        question_word_idx = np.concatenate((self.question_sorted[idx], question_pad_word))

        #We need to return the character mapping also
        context_raw = self.context_raw_sorted[idx]
        question_raw = self.question_raw_sorted[idx]

        ctx4char = context_raw.split()
        ctx4char = [word2charix(word, self.char2ix) for word in ctx4char]
        ctx4char = np.array(ctx4char)

        q4char = question_raw.split()
        q4char = [word2charix(word, self.char2ix) for word in q4char]
        q4char = np.array(q4char)

        #Pad character embedding
        context_pad_char = np.zeros((constants.MAX_CONTEXT - ctx4char.shape[0], ctx4char.shape[1]))
        question_pad_char = np.zeros((self.max_question_len - q4char.shape[0], q4char.shape[1]))

        context_char_idx = np.concatenate((ctx4char, context_pad_char))
        question_char_idx =  np.concatenate((q4char, question_pad_char))

        del context_pad_word
        del question_pad_word
        del context_pad_char
        del question_pad_char
        del ctx4char, q4char

        return context_word_idx, question_word_idx, context_char_idx, question_char_idx, self.spans_sorted[idx], context_raw, question_raw