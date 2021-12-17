import numpy as np

# TODO: change this initialization to parameters
develop_file_path = "../develop.txt"
topics_file_path = "../topics.txt"

# CONST
RARE_WORDS_THRESHOLD = 3
CLUSTER_NUMBERS = 9

# TODO: When we will finish the implementation we need to normalize those arguments

EPSILON = 0.000001
LAMBDA_VALUE = 2


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
            del self.word_to_freq[key]

    def __len__(self):
        # TODO: check if length of document (n_t) is number of unique words or total number of words!
        return len(self.word_to_freq)

    def __getitem__(self, word):
        if word not in self.word_to_freq.keys():
            return 0
        return self.word_to_freq[word]

    def keys(self):
        return self.word_to_freq.keys()


class EM(object):
    def __init__(self, documents, all_develop_documents, clusters_number=9, epsilon=EPSILON, lambda_value=2):
        self.clusters_number = clusters_number
        self.number_of_documents = len(documents)
        self.word2k = {word: k for k, word in enumerate(all_develop_documents.keys())}
        self.document2t = {document: t for t, document in enumerate(documents)}
        self.vocabulary_size = len(all_develop_documents)
        self.p_i_k = np.zeros((self.clusters_number, self.vocabulary_size))
        self.alpha = np.zeros((self.clusters_number, 1))
        self.epsilon = epsilon
        self.lambda_value = lambda_value

        # n_t_k - frequency of the word k in document t
        self.n_t_k = np.zeros((self.number_of_documents, self.vocabulary_size))
        # dictionary represents length of document t
        self.nt = np.zeros((self.number_of_documents, 1))
        for document in self.document2t:
            t = self.document2t[document]
            for word in self.word2k:
                self.n_t_k[t][self.word2k[word]] = document[word]
            self.nt[t] = len(document)

        # initializing the model
        self.w_t_i = np.zeros((self.number_of_documents, self.clusters_number))
        # 1st to cluster 1, 2nd to cluster 2 ... idx % cluster number initialization
        for i in range(self.number_of_documents):
            self.w_t_i[i][i % self.clusters_number] = 1

        # do m_step
        self.m_step()

    def m_step(self):
        # alpha computation

        self.alpha = np.zeros((self.clusters_number, 1))
        # summing by column (column is classification)
        self.alpha = np.sum(self.w_t_i, axis=0) / self.number_of_documents
        self.alpha = np.maximum(self.alpha, self.epsilon)
        # if we changed to at least one of the alphas we need to normalize in order it to be summed up to 1
        self.alpha /= np.sum(self.alpha)

        # p_i_k is the probability for word k when we assume we re in category i : p(w_k | c_i)
        numerator = np.dot(self.w_t_i.T, self.n_t_k) + self.lambda_value  # -> output size |cluster| X |vocab|
        denominator = np.dot(self.w_t_i.T,
                             self.nt) + self.vocabulary_size * self.lambda_value  # -> output size |cluster| x |1|
        self.p_i_k = numerator / denominator

    def z(self):
        logged_alpha = np.log(self.alpha)  # |cluster| x 1
        right_hand = np.dot(self.n_t_k, np.log(self.p_i_k.T))  # |document| x |cluster|
        left_hand = np.broadcast_to(logged_alpha, (self.number_of_documents, self.clusters_number))
        self.z = right_hand + left_hand  # |document| x |cluster|

    def m(self):
        # find max z for each classification
        self.m = np.max(self.z, axis=1)  # |document| x |1|
        print(self.m)

    def e_step(self):
        self.z()
        self.m()

    def get_p_i_k(self):
        return self.p_i_k

    def get_alpha(self):
        return self.alpha


def extract_topics(topics_file_path):
    topics = set()
    with open(topics_file_path, "r") as topics_file:
        for line in topics_file.readlines():
            striped_line = line.strip()
            if 0 != len(striped_line):
                topics.add(striped_line)
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


def run():
    topics, documents, all_develop_documents = initialization_process(develop_file_path, topics_file_path)
    em_model = EM(documents=documents, all_develop_documents=all_develop_documents, clusters_number=CLUSTER_NUMBERS,
                  epsilon=EPSILON, lambda_value=LAMBDA_VALUE)
    # print(np.max(em_model.p_i_k))
    em_model.e_step()


if __name__ == "__main__":
    run()
