### TODO: change this initaliation to parameters
develop_file_path = "../develop.txt"
topics_file_path = "../topics.txt"

RARE_WORDS_THRESHOLD = 3


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
    print(topics)


if __name__ == "__main__":
    run()
