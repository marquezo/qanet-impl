from __future__ import print_function
import json
import nltk
import os
from tqdm import tqdm
import sys

def data_from_json(filename):
    with open(filename) as data_file:
        data = json.load(data_file)
    return data


# Input is the output of calling nltk.word_tokenize(.)
def tokenize(words_tokenized):
    tokens = [token.replace("``", '"').replace("''", '"') for token in words_tokenized]
    return tokens


# Return a map of the form {0: ['Architecturally', 0], 15: [',', 1], 17: ['the', 2]}
# where the index is the character position where the token starts (taking into account whitespace)
# and the value is a list of two elements: token, token id
# Used to know where the answer starts and ends in the context paragraph
def token_idx_map(context, context_tokens):
    acc = ''
    current_token_idx = 0
    token_map = dict()

    for char_idx, char in enumerate(context):

        if char != u' ':
            acc += char
            context_token = context_tokens[current_token_idx]  # str(context_tokens[current_token_idx], 'utf-8')

            if acc == context_token:
                syn_start = char_idx - len(acc) + 1
                token_map[syn_start] = [acc, current_token_idx]
                acc = ''
                current_token_idx += 1
    return token_map


def read_write_dataset(dataset, tier, prefix, max_context_length=400):
    """Reads the dataset, extracts context, question, answer,
    and answer pointer in their own file. Returns the number
    of questions and answers processed for the dataset"""
    qn, an = 0, 0
    num_errors_parsing = 0
    num_context_skipped = 0

    with open(os.path.join(prefix, tier + '.context'), 'w') as context_file, \
            open(os.path.join(prefix, tier + '.question'), 'w') as question_file, \
            open(os.path.join(prefix, tier + '.answer'), 'w') as text_file, \
            open(os.path.join(prefix, tier + '.span'), 'w') as span_file:

        for articles_id in tqdm(range(len(dataset['data'])), desc="Preprocessing {}".format(tier)):
            article_paragraphs = dataset['data'][articles_id]['paragraphs']
            for pid in range(len(article_paragraphs)):
                context = article_paragraphs[pid]['context']

                # The following replacements are suggested in the paper
                # BidAF (Seo et al., 2016)
                context = context.replace("''", '" ')
                context = context.replace("``", '" ')

                # Use NLTK to tokenize the words in the context
                words_tokenized = nltk.word_tokenize(context)

                if (len(words_tokenized) < max_context_length):

                    context_tokens = tokenize(words_tokenized)

                    answer_map = token_idx_map(context, context_tokens)

                    qas = article_paragraphs[pid]['qas']

                    # Go through every question belonging to the current context
                    for qid in range(len(qas)):
                        question = qas[qid]['question']

                        question_tokens = nltk.word_tokenize(question)
                        question_tokens = tokenize(question_tokens)

                        answers = qas[qid]['answers']
                        qn += 1

                        # Only one answer in training set. TODO in dev set?
                        num_answers = range(1)

                        for ans_id in num_answers:

                            # it contains answer_start, text
                            text = qas[qid]['answers'][ans_id]['text']

                            text_tokens = nltk.word_tokenize(text)
                            text_tokens = tokenize(text_tokens)

                            answer_start = qas[qid]['answers'][ans_id]['answer_start']
                            answer_end = answer_start + len(text)

                            # Get length of last word in the answer to be able to identify
                            # the token in the answer_map
                            last_word_answer = len(text_tokens[-1])

                            # print (answer_start, answer_end, last_word_answer)

                            try:
                                # answer start token index - in the map of the context previously created
                                a_start_idx = answer_map[answer_start][1]
                                # answer end token index - in the map of the context previously created
                                a_end_idx = answer_map[answer_end - last_word_answer][1]

                                context_file.write(' '.join(context_tokens) + '\n')
                                question_file.write(' '.join(question_tokens) + '\n')
                                text_file.write(' '.join(text_tokens) + '\n')
                                # Include the start token index for the span as well as the end token index
                                span_file.write(' '.join([str(a_start_idx), str(a_end_idx)]) + '\n')
                                # span_file.write(' '.join([str(answer_start), str(answer_end)]) + '\n')

                            except Exception as e:
                                #                                 print (qas)
                                #                                 print (answer_map)
                                #                                 print ('---')
                                #                                 print (question_tokens)
                                #                                 print ('---')
                                #                                 print (text_tokens)
                                #                                 print ('---')
                                #                                 print (' '.join([str(a_start_idx), str(a_end_idx)]) + '\n')
                                #                                 print (e)
                                #                                 return None
                                num_errors_parsing += 1

                            an += 1
                else:
                    num_context_skipped +=1
                    #print("Context is too long")

    print("Contexts discarded because too long: {} | Question/answer pairs ignored because parsing errors: {}".format(num_context_skipped, num_errors_parsing))
    return qn, an

def read_write_dev_dataset(dataset, tier, prefix, max_context_length=400):
    """Reads the dataset, extracts context, question, answer,
    and answer pointer in their own file. Returns the number
    of questions and answers processed for the dataset"""
    qn, an = 0, 0
    num_errors_parsing = 0
    num_context_skipped = 0

    with open(os.path.join(prefix, tier + '.context'), 'w') as context_file, \
            open(os.path.join(prefix, tier + '.question'), 'w') as question_file, \
            open(os.path.join(prefix, tier + '.answer'), 'w') as text_file, \
            open(os.path.join(prefix, tier + '.span'), 'w') as span_file:

        for articles_id in tqdm(range(len(dataset['data'])), desc="Preprocessing {}".format(tier)):
            article_paragraphs = dataset['data'][articles_id]['paragraphs']
            for pid in range(len(article_paragraphs)):
                context = article_paragraphs[pid]['context']

                # The following replacements are suggested in the paper
                # BidAF (Seo et al., 2016)
                context = context.replace("''", '" ')
                context = context.replace("``", '" ')

                # Use NLTK to tokenize the words in the context
                words_tokenized = nltk.word_tokenize(context)

                if (len(words_tokenized) < max_context_length):

                    context_tokens = tokenize(words_tokenized)

                    answer_map = token_idx_map(context, context_tokens)

                    qas = article_paragraphs[pid]['qas']

                    # Go through every question belonging to the current context
                    for qid in range(len(qas)):
                        question = qas[qid]['question']

                        question_tokens = nltk.word_tokenize(question)
                        question_tokens = tokenize(question_tokens)

                        answers = qas[qid]['answers']
                        qn += 1
                        
                        text_tokens = []
                        ans_start = []
                        ans_end = []
                        ans_last_word = []
                        
                        num_answers = len(answers)
                        

                        for ans_id in range(num_answers):

                            # it contains answer_start, text
                            text = qas[qid]['answers'][ans_id]['text']

                            text_tokens_tmp = nltk.word_tokenize(text)
                            text_tokens.append(tokenize(text_tokens_tmp))

                            answer_start = qas[qid]['answers'][ans_id]['answer_start']
                            answer_end = answer_start + len(text)
                            
                            ans_start.append(answer_start)
                            ans_end.append(answer_end)

                            # Get length of last word in the answer to be able to identify
                            # the token in the answer_map
                            last_word_answer = len(text_tokens_tmp[-1])
                            ans_last_word.append(last_word_answer)

                            # print (answer_start, answer_end, last_word_answer)

                        spans = []
                        

                        try:
                            for i in range(num_answers):
                                # answer start token index - in the map of the context previously created
                                a_start_idx = answer_map[ans_start[i]][1]
#                                print(a_start_idx)
                                # answer end token index - in the map of the context previously created
                                a_end_idx = answer_map[ans_end[i] - ans_last_word[i]][1] 
                                spans.append([a_start_idx, a_end_idx])
                           
#                            print(text_tokens)
                            
                            context_file.write(' '.join(context_tokens) + '\n')
                            question_file.write(' '.join(question_tokens) + '\n')
#                            text_file.write(' '.join(elem for elem in text_tokens) + '\n')
                            # Include the start token index for the span as well as the end token index
                            
                            span_to_write = ""
                            
                            for span in spans:
                                span_to_write += (str(span[0]) + " " + str(span[1]) + " ")
                            
#                            print("to write", span_to_write)
                            
#                           span_file.write(' '.join([str(spans[i,0]), str(spans[i,1]) for i in range(len(spans))]) + '\n')
                            span_file.write(span_to_write + "\n")
                                                        # span_file.write(' '.join([str(answer_start), str(answer_end)]) + '\n')                            
                                
                        
                            
                        except Exception as e:
#                            print(a_start_idx)
#                            print(ans_end)
#                            print(ans_last_word)
#                            print (qas)
#                            print (answer_map)
#                            print ('---')
#                            print (question_tokens)
#                            print ('---')
#                            print (text_tokens)
#                            print ('---')
#                            #print (' '.join([str(a_start_idx), str(a_end_idx)]) + '\n')
#                            print ("Exception", e)
#                            return None
                            num_errors_parsing += 1

                        an += 1
                else:
                    num_context_skipped +=1
                    #print("Context is too long")

    print("Contexts discarded because too long: {} | Question/answer pairs ignored because parsing errors: {}".format(num_context_skipped, num_errors_parsing))
    return qn, an


if __name__ == "__main__":
    
    dataset_file = 'dev-v1.1.json'
    with open(dataset_file) as f:
        dataset = json.load(f)
    
    read_write_dev_dataset(dataset, 'devtest', 'data')