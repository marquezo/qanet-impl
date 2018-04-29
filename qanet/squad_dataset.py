from torch.utils.data import Dataset
import numpy as np
import constants
import json
from vocab_util import word2charix

# Return (context, question) pairs sorted by length of context in terms of tokens
# (i.e., first return longer contexts)
class SquadDataset(Dataset):

    def __init__(self, file_ids_ctx, file_ids_q, file_ctx, file_q, char2ix_file):

        #This will be used in the character mapping
        with open(char2ix_file) as json_data:
            self.char2ix = json.load(json_data)

        list_context = []
        list_questions = []
        self.max_question_len = 0
        len_contexts = []

        list_context_raw = []
        list_question_raw = []

        with open(file_ids_q, "r") as question_file_ids, \
                open(file_ids_ctx, "r") as context_file_ids, \
                open(file_q, "r") as question_file, \
                open(file_ctx, "r") as context_file:

            # We need to find the max token length for questions
            # Also, collect the contexts and questions to sort them acc. to context length
            # (length in terms of tokens, NOT in terms of characters)
            for context, question, context_raw, question_raw in \
                    zip(context_file_ids, question_file_ids, context_file, question_file):

                context_array = np.fromstring(context.strip('\n'), dtype=int, sep=' ')
                question_array = np.fromstring(question.strip('\n'), dtype=int, sep=' ')

                list_context_raw.append(context_raw.strip('\n'))
                list_question_raw.append(question_raw.strip('\n'))

                list_context.append(context_array)
                list_questions.append(question_array)

                if question_array.shape[0] > self.max_question_len:
                    self.max_question_len = question_array.shape[0]

                len_contexts.append(context_array.shape[0])

        pos_for_batch = np.argsort(len_contexts)

        self.context_sorted = []
        self.question_sorted = []
        self.context_raw_sorted = []
        self.question_raw_sorted = []

        for i in pos_for_batch:
            self.context_sorted.append(list_context[i])
            self.question_sorted.append(list_questions[i])
            self.context_raw_sorted.append(list_context_raw[i])
            self.question_raw_sorted.append(list_question_raw[i])

    def __len__(self):
        return len(self.context_sorted)

    def __getitem__(self, idx):

        context_pad = np.array([constants.PAD_ID] * (constants.MAX_CONTEXT - self.context_sorted[idx].shape[0]))
        question_pad = np.array([constants.PAD_ID] * (self.max_question_len - self.question_sorted[idx].shape[0]))

        context_to_return = np.concatenate((self.context_sorted[idx], context_pad))
        question_to_return = np.concatenate((self.question_sorted[idx], question_pad))

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
        pad4ctx = np.zeros((constants.MAX_CONTEXT - ctx4char.shape[0], ctx4char.shape[1]))
        pad4q = np.zeros((self.max_question_len - q4char.shape[0], q4char.shape[1]))

        char_mapping_context = np.concatenate((ctx4char, pad4ctx))
        char_mapping_question =  np.concatenate((q4char, pad4q))

        return context_to_return, question_to_return, char_mapping_context, char_mapping_question