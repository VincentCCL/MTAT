import sentencepiece as spm
import sys

infile=sys.argv[1]
prefix=sys.argv[2]
try: vocab_size=sys.argv[3]
except: vocab_size=8000

spm.SentencePieceTrainer.train(input=infile, model_prefix=prefix, vocab_size=vocab_size, train_extremely_large_corpus=True)
