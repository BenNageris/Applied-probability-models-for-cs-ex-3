import math
from collections import Counter

import numpy as np

# TODO: change this initialization to parameters
develop_file_path = "../develop.txt"
topics_file_path = "../topics.txt"

# CONST
RARE_WORDS_THRESHOLD = 3
CLUSTER_NUMBERS = 9
K = 10

# TODO: When we will finish the implementation we need to normalize those arguments

EPSILON = 0.000000000001
LAMBDA_VALUE = 0.965


class Document(object):
    def __init__(self, topics):
        self.word_to_freq = {}
        self.topics = topics

    def add(self, word):
        if word in self.word_to_freq:
            self.word_to_freq[word] = self.word_to_freq[word] + 1
        else:
            self.word_to_freq[word] = 1

    def remove(self, key):
        if key in self.word_to_freq:
            self.word_to_freq.pop(key)

    def __len__(self):
        #return len(self.word_to_freq)
        sum = 0
        for freq in self.word_to_freq.values():
            sum = sum + freq
        return sum

    def __getitem__(self, word):
        if word not in self.word_to_freq.keys():
            self.word_to_freq[word] = 0
        return self.word_to_freq[word]

    def keys(self):
        return self.word_to_freq.keys()

    def len_keys(self):
        return len(self.word_to_freq)


class EM(object):
    def __init__(self, documnets,voc,number_of_clusters):
        self.number_of_words_in_vocab = len(voc)
        self.documents_number=len(documnets)
        self.to_index={word: i for i, word in enumerate(voc.keys())}
        self.documents=documnets
        self.number_of_clasters=number_of_clusters
        self.ntk = np.zeros((self.documents_number,self.number_of_words_in_vocab))
        self.nt = np.zeros((self.documents_number,1))
        self.init_nt_ntk()
        self.wti = np.zeros((self.documents_number, self.number_of_clasters))
        self.pki = np.zeros((self.number_of_words_in_vocab, self.number_of_clasters))
        self.alpha = np.zeros((self.number_of_clasters, 1))
        self.m = None
        self.z = None
        self.init_wti()
        self.epsilon = np.array([EPSILON])
        self.m_step()
    def init_nt_ntk(self):
        for i, doc in enumerate(self.documents):
            nt = len(doc.keys())
            self.nt[i]=nt
            for event in doc.keys():
                j=self.to_index[event]
                self.ntk[i][j]=doc[event]
    def init_wti(self):
        for i in range(self.documents_number):
            self.wti[i][i % self.number_of_clasters] = 1
    def m_step(self):
        self.alpha = np.sum(self.wti, axis=0)/self.documents_number
        self.alpha = np.maximum(self.alpha, self.epsilon)
        self.alpha /= np.sum(self.alpha)
        self.alpha = self.alpha.reshape(self.alpha.shape[0], 1)
        up = (np.dot(self.ntk.T,self.wti)+LAMBDA_VALUE)
        print(up.shape)
        down = (np.dot(self.nt.T, self.wti)+self.number_of_words_in_vocab*LAMBDA_VALUE)
        print(down.shape)
        self.pki = up/down

    def e_step(self):
        self.z = np.broadcast_to(self.alpha,(self.number_of_clasters, self.documents_number)).T\
                 +np.dot(self.ntk,np.log(self.pki))
        self.m = np.max(self.z, axis=1).reshape((self.documents_number, 1))
        exceeded_limit = self.z - self.m < -K
        for i in range(self.documents_number):
            for j in range(self.number_of_clasters):
                diff=self.z[i][j] - self.m[i]
                if diff<-K:
                    self.wti[i][j] = 0
                else:
                    self.wti[i][j] =np.exp(diff)
            #NOTE here is different
            self.wti[i] / np.sum(self.wti[i])
        return self.wti
    # def log_likelihood(self):
    #     boolean_table = self.z_matrix - self.m_vector >= (-1) * self.k
    #     array = np.zeros((self.number_of_documents, 1))
    #     for t in range(self.number_of_documents):
    #         for i in range(self.clusters_number):
    #             if boolean_table[t][i]:
    #                 array[t] += np.exp(self.z_matrix[t][i] - self.m_vector[t])
    #     return np.sum(np.log(array) + self.m_vector)
    #
    # def perplexity(self):
    #     log_likelihood = self.log_likelihood()
    #     return np.power(2, -1 * log_likelihood / np.sum(self.nt))
    #
    # def confusion_matrix(self):
    #     document_cluster = np.argmax(self.w_t_i, axis=1)
    #     confusion_matrix_ = np.zeros((self.clusters_number, self.clusters_number + 1))
    #     for document in self.document2t:
    #         t = self.document2t[document]
    #         i = document_cluster[t]
    #         for topic in document.topics:
    #             j = self.topic2index[topic]
    #             confusion_matrix_[i][j] += 1
    #             confusion_matrix_[i][self.clusters_number] += 1
    #     return confusion_matrix_[confusion_matrix_[:, self.clusters_number].argsort()[::-1]]
    #
    # def print_confusion_matrix(self, topics):
    #     confusion_matrix_ = self.confusion_matrix()
    #     print(topics)
    #     print(confusion_matrix_)
    #
    # def get_p_i_k(self):
    #     return self.p_i_k
    #
    # def get_alpha(self):
    #     return self.alpha

    def train(self,ep=10):
        print(self.accuracy())
        for _ in range(ep):
            self.e_step()
            self.m_step()
            print(self.accuracy())
        # prev_prep = float('inf')
        # cur_prep = float('inf')
        # while prev_prep - cur_prep > self.epsilon or cur_prep == float('inf'):
        #     self.e_step()
        #     self.m_step()
        #     print("acc",self.accuracy())
        #     #print("likelihood",self.log_likelihood())
        #     prev_prep = cur_prep
        #     #cur_prep = self.perplexity()
        #     print("cur_prep",cur_prep)
        #
    def count_topics_in_cluster(self, document2cluster):
        cluster2topics = {i: Counter() for i in range(self.number_of_clasters)}
        for t, document in enumerate(self.documents):
            cluster = document2cluster[t]
            cluster2topics[cluster].update(document.topics)
        return cluster2topics

    def find_cluster2topic(self, document2cluster):
        output = {i: 0 for i in range(self.number_of_clasters)}
        cluster2topics = self.count_topics_in_cluster(document2cluster)

        def topicOrNone(topic_in_list):
            if len(topic_in_list) == 0:
                return 'None'
            return topic_in_list[0][0]

        output.update({i: topicOrNone(counter.most_common(1)) for i, counter in
                       zip(cluster2topics.keys(), cluster2topics.values())})
        return output

    def accuracy(self):
        document2cluster = np.argmax(self.wti, axis=1)
        cluster2topic = self.find_cluster2topic(document2cluster)
        hit=0
        for i,doc in enumerate(self.documents):
            clust=document2cluster[i]
            topic=cluster2topic[clust]
            hit +=topic in doc.topics
        return hit/self.documents_number

    # def accuracy1(self):
    #     #print(self.w_t_i)
    #     document2classification = self.wti.argmax(axis=1)
    #     #print(document2classification)
    #     idx = 0
    #     hit = 0
    #     for i in range(len(self.document_list)):
    #        topics= self.document_list[i].topics
    #        topics_idx=[self.topic2index[topic] for topic in topics]
    #        #print(topics)
    #        #print(topics_idx)
    #        #print(document2classification[i])
    #        if document2classification[i] in topics_idx:
    #            hit=hit+1
    #        idx=idx+1
    #     #print(self.document_list[0].topics)
    #     #print("hit",hit)
    #     #print("idx",idx)
    #     return hit/idx
    #
    # def accuracy2(self):
    #     document2classification = self.w_t_i.argmax(axis=1)
    #     print(document2classification.shape, np.max(document2classification), np.min(document2classification))
    #     idx=0
    #     cnt = 0
    #     print(self.topic2index.items())
    #
    #     for document, t in self.document2t.items():
    #         document_topics = [self.topic2index[topic] for topic in document.topics]
    #         # print(document2classification[t], document_topics)
    #         if document2classification[t] in document_topics:
    #             cnt += 1
    #         print("idx", idx)
    #         print("cnt", cnt)
    #         print("document2classification[t]",document2classification[t])
    #         print("document_topics",document_topics)
    #         print("pred",document2classification[idx])
    #         print("real",document.topics)
    #         idx=idx+1
    #     return cnt / self.number_of_documents


def extract_topics_old(topics_file_path):
    topics = []
    with open(topics_file_path, "r") as topics_file:
        for line in topics_file.readlines():
            striped_line = line.strip()
            if 0 != len(striped_line) and striped_line not in topics:
                topics.append(striped_line)
    return topics


def extract_topic_from_line(content):
    return content.replace("<", "").replace(">", "").split("\t")[2:]


def read_develop_file(develop_file_path):
    documents = []
    all_develop_documents = Document(topics=[])
    with open(develop_file_path) as develop_file:
        lines = develop_file.readlines()
    striped_lines = [line.strip() for line in lines if line.strip()]
    topic_content_zip = zip(striped_lines[0::2], striped_lines[1::2])
    for topics, content in topic_content_zip:
        extracted_topics = extract_topic_from_line(topics)
        document = Document(topics=extracted_topics)
        for word in content.split():
            document.add(word=word)
            all_develop_documents.add(word=word)
        documents.append(document)
    return documents, all_develop_documents


def initialization_process(develop_file_path, topics_file_path):
    topics = extract_topics(topics_file_path)
    documents, all_develop_documents = read_develop_file(develop_file_path)
    rare_words = [word for word in all_develop_documents.word_to_freq if
                  all_develop_documents.word_to_freq[word] <= RARE_WORDS_THRESHOLD]
    for rare_word in rare_words:
        all_develop_documents.remove(rare_word)
        for document in documents:
            document.remove(rare_word)
    return topics, documents, all_develop_documents


def run2():
    topics, documents, all_develop_documents = initialization_process(develop_file_path, topics_file_path)
    print(topics)
    em_model = EM(documents=documents, all_develop_documents=all_develop_documents, topics=topics,
                  clusters_number=CLUSTER_NUMBERS,
                  epsilon=EPSILON, lambda_value=LAMBDA_VALUE, k=K)
    em_model.train()
    # em_model.print_confusion_matrix(topics)
    print(em_model.accuracy())

#utils start
RARE_THRESHOLD = 3
TOPICS_INDEX = 2
TOPICS = ['acq',
          'money-fx',
          'grain',
          'crude',
          'trade',
          'interest',
          'ship',
          'wheat',
          'corn']


class Document(object):
    def __init__(self, dict_to_wrap=None, topics=None):
        super().__init__()
        if dict_to_wrap is None:
            dict_to_wrap = {}
        if topics is None:
            topics = []
        self.wraped_dict = dict_to_wrap
        self.topics = topics

    def __getitem__(self, item):
        if item not in self.wraped_dict.keys():
            self.wraped_dict[item] = 0
        return self.wraped_dict[item]

    def __setitem__(self, key, value):
        self.wraped_dict[key] = value

    def __len__(self):
        return len(self.wraped_dict.keys())
        total = 0
        for k in self.wraped_dict.keys():
            total = total+self.wraped_dict[k]
        return total

    def __repr__(self):
        return self.wraped_dict.__repr__()

    def keys(self):
        return self.wraped_dict.keys()

    def values(self):
        return self.wraped_dict.values()

    def remove(self, key):
        if key in self.wraped_dict:
            del self.wraped_dict[key]


def checkline(line):
    return line.startswith("<") and line.endswith(">")


def extract_topics(line):
    return line[:-1].split()[TOPICS_INDEX:]


def read_dev_file(vocab_path):
    samples = []
    events = Document()
    with open(vocab_path) as vocab_file:
        line = vocab_file.readline()
        topics = []
        while line:
            line = line.strip()
            if not (checkline(line) or line == ""):
                sample = Document(topics=topics)
                for word in line.split():
                    sample[word] += 1
                    events[word] += 1
                samples.append(sample)
            elif line != "":
                topics = extract_topics(line)
            line = vocab_file.readline()
        vocab_file.close()
    return samples, events


def get_topics(topic_path):
    try:
        with open(topic_path) as topic_file:
            topics = []
            line = topic_file.readline()
            while line:
                line = line.strip()
                if line != "":
                    topics.append(line)
                line = topic_file.readline()
        topic_file.close()
        return topics
    except IOError:
        return TOPICS


def prep_data(path):
    samples, events = read_dev_file(path)
    rare_words = [word for word in events.keys() if events[word] <= RARE_THRESHOLD]
    for word in rare_words:
        events.remove(word)
        for sample in samples:
            sample.remove(word)
    return samples, events

#utils end
def run():
    cluster_num=9
    samples, events = prep_data('../develop.txt')
    topics = get_topics('../topics.txt')
    topic2index = {topic: i for i, topic in enumerate(topics)}
    model = EM(samples, events, cluster_num)
    model.train()
    pass
if __name__ == "__main__":
    run()
