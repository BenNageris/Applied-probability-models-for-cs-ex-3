# TODO: change this initialization to parameters
develop_file_path = "../develop.txt"
topics_file_path = "../topics.txt"

# CONST
RARE_WORDS_THRESHOLD = 3

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
        self.clusters = list(range(0, clusters_number))
        self.vocabulary_size = len(all_develop_documents)
        # dictionary where key is: (word_k,document_y) = frequency of word k in document y
        self.n_t_k = {}
        # dictionary represents length of document t
        self.nt = {}
        for document in self.document2t:
            t = self.document2t[document]
            for word in self.word2k:
                self.n_t_k[(t, self.word2k[word])] = document[word]
            self.nt[t] = len(document)

        # initializing the model
        self.w_t_i = {}
        for i in range(self.number_of_documents):
            for j in self.clusters:
                self.w_t_i[i, j] = 0
        # 1st to cluster 1, 2nd to cluster 2 ... idx % cluster number initialization
        for i in range(self.number_of_documents):
            self.w_t_i[i, i % self.clusters_number] = 1

        self.p_i_k = {}
        self.alpha = [0] * self.clusters_number
        self.epsilon = epsilon
        self.lambda_value = lambda_value

        # do m_step
        self.m_step()

    def m_step(self):
        # alpha computation
        self.alpha = [0] * self.clusters_number
        for t, cluster_idx in self.w_t_i:
            self.alpha[cluster_idx] += self.w_t_i[t, cluster_idx]
        self.alpha = [alpha / self.number_of_documents for alpha in self.alpha]
        self.alpha = [max(self.epsilon, alpha) for alpha in self.alpha]
        alpha_sum = sum(self.alpha)
        self.alpha = [alpha / alpha_sum for alpha in self.alpha]

        # P_i_k computation
        # p_i_k is the probability for word k when we assume we re in category i : p(w_k | c_i)
        for word in self.word2k:
            k = self.word2k[word]
            for category_idx in self.clusters:
                numerator = 0
                denominator = 0
                for t in range(self.number_of_documents):
                    numerator += self.w_t_i[t, category_idx] * self.n_t_k[t, k]
                    denominator += self.w_t_i[t, category_idx] * self.nt[t]
                smooth_p_i_k = (numerator + self.lambda_value) / (
                        denominator + self.vocabulary_size * self.lambda_value)
                self.p_i_k[category_idx, k] = smooth_p_i_k

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
    print(em_model.p_i_k)


if __name__ == "__main__":
    run()
CLUSTER_NUMBERS = 9
