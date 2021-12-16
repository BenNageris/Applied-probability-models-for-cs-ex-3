### TODO: change this initaliation to parameters
develop_file_path = "../develop.txt"
topics_file_path = "../topics.txt"

RARE_WORDS_THRESHOLD = 3
CLUSTER_NUMBERS = 9


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
        return len(self.word_to_freq)

    def __getitem__(self, word):
        if word not in self.word_to_freq.keys():
            return 0
        return self.word_to_freq[word]

    def keys(self):
        return self.word_to_freq.keys()


class EM(object):
    def __init__(self, documents, all_develop_documents, clusters_number=9):
        self.clusters_number = clusters_number
        self.number_of_documents = len(documents)
        self.word2k = {word: k for k, word in enumerate(all_develop_documents.keys())}
        self.document2t = {document: t for t, document in enumerate(documents)}

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
        self.w_ti = {}
        for i in range(self.number_of_documents):
            self.w_ti[i, i % self.clusters_number] = 1
        


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


def initilaization_preocess(develop_file_path, topics_file_path):
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
    topics, documents, all_develop_documents = initilaization_preocess(develop_file_path, topics_file_path)
    em_model = EM(documents=documents, all_develop_documents=all_develop_documents, clusters_number=CLUSTER_NUMBERS)


if __name__ == "__main__":
    run()
