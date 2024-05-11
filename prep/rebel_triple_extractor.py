import os
import shutil
import spacy

from llama_index.core import SimpleDirectoryReader
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.core.schema import TransformComponent
from transformers import pipeline
from typing import List

from prep_utils import extract_chapter_from_fn


DATA_DIR = "../data"
CHAPTERS_DIR = os.path.join(DATA_DIR, "chapters")
REBEL_CHUNKS_DIR = os.path.join(DATA_DIR, "rebel-chunks")
REBEL_MODEL_NAME = "Babelscape/rebel-large"


class ChunkTextToSentencesSplitter(TransformComponent):

    _spacy_lm: spacy.language.Language = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._spacy_lm = kwargs.get("spacy_lm")

    def __call__(self, nodes: List[TextNode], **kwargs):
        # output_nodes = []
        num_sents_in_node = 0
        for node in nodes:
            sents = [sent.text for sent in self._spacy_lm(node.text).sents]
            node.metadata["sentences"] = sents
            num_sents_in_node += len(sents)
            # output_nodes.append(node)
        print(f"... split nodes into {num_sents_in_node} sentences")
        # return output_nodes
        return nodes


class RebelTripleExtractor(TransformComponent):

    _model: pipeline = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model = kwargs.get("triple_extractor_model")

    def _extract_triplets(self, input_text, decoded_text):
        triplets = []
        relation, subject, relation, object_ = "", "", "", ""
        decoded_text = decoded_text.strip()
        current = "x"
        for token in decoded_text.replace("<s>", "").replace(
                "<pad>", "").replace("</s>", "").split():
            if token == "<triplet>":
                current = "t"
                if relation != "":
                    triplets.append({
                        "head": subject.strip(),
                        "type": relation.strip(),
                        "tail": object_.strip()
                    })
                    relation = ""
                subject = ""
            elif token == "<subj>":
                current = "s"
                if relation != "":
                    triplets.append({
                        "head": subject.strip(),
                        "type": relation.strip(),
                        "tail": object_.strip()
                    })
                object_ = ""
            elif token == "<obj>":
                current = "o"
                relation = ""
            else:
                if current == "t":
                    subject += " " + token
                elif current == "s":
                    object_ += " " + token
                elif current == "o":
                    relation += " " + token
        if subject != "" and subject in input_text and \
                relation != "" and relation in input_text and \
                object_ != "" and object_ in input_text:
            triplets.append({
                "head": subject.strip(),
                "type": relation.strip(),
                "tail": object_.strip()
            })
        return triplets

    def __call__(self, nodes: List[TextNode], **kwargs):
        chunk_idxs, sentences = [], []
        for idx, node in enumerate(nodes):
            node_sents = node.metadata["sentences"]
            sentences.extend(node_sents)
            chunk_idxs.extend([idx + 1] * len(node_sents))
        # send the full batch of sentences (per chapter) to the model to process
        print("... generating triples", end="")
        outputs = self._model(sentences,
                              return_tensors=True,
                              return_text=False)
        token_ids = [output["generated_token_ids"] for output in outputs]
        decoded_tokens = self._model.tokenizer.batch_decode(token_ids)
        chunkid_to_triples = {}
        for chunk_idx, sentence, decoded in zip(
                chunk_idxs, sentences, decoded_tokens):
            triplets = self._extract_triplets(sentence, decoded)
            if chunk_idx not in chunkid_to_triples.keys():
                chunkid_to_triples[chunk_idx] = triplets
            else:
                chunkid_to_triples[chunk_idx].extend(triplets)
        # now set the triples back into the chunks
        output_nodes = []
        num_triples_in_node = 0
        for chunk_idx, node in enumerate(nodes):
            triplets = chunkid_to_triples.get(chunk_idx + 1, [])
            if len(triplets) > 0:
                node.metadata["triplets"] = triplets
                num_triples_in_node += len(triplets)
            output_nodes.append(node)
        print(f": {num_triples_in_node} triples found")
        return output_nodes


if True:

    text_splitter = SentenceSplitter(
        chunk_size=500, chunk_overlap=100, paragraph_separator="\n\n")

    spacy_lm = spacy.load("en_core_web_sm")
    chunk_sentence_splitter = ChunkTextToSentencesSplitter(spacy_lm=spacy_lm)

    triple_extractor_model = pipeline(
        "text2text-generation",
        model=REBEL_MODEL_NAME,
        tokenizer=REBEL_MODEL_NAME,
        device="cuda:0")
    triple_extractor_model._batch_size = 8
    triple_extractor = RebelTripleExtractor(
        triple_extractor_model=triple_extractor_model)

    pipeline_transformations = [
        text_splitter,
        chunk_sentence_splitter,
        triple_extractor
    ]

    ingestion_pipeline = IngestionPipeline(
        transformations=pipeline_transformations)

    shutil.rmtree(REBEL_CHUNKS_DIR, ignore_errors=True)
    os.makedirs(REBEL_CHUNKS_DIR, exist_ok=True)

    for chapter_fn in os.listdir(CHAPTERS_DIR):
        print("processing file:", chapter_fn)
        chapter_file = os.path.join(CHAPTERS_DIR, chapter_fn)
        docs = SimpleDirectoryReader(input_files=[chapter_file]).load_data()
        nodes = ingestion_pipeline.run(documents=docs)
        for cid, node in enumerate(nodes):
            chapter_id = extract_chapter_from_fn(chapter_fn)
            chunk_fn = os.path.join(
                REBEL_CHUNKS_DIR, f"chunk-{chapter_id}-{cid}.json")
            with open(chunk_fn, "w") as f:
                f.write(node.json())
