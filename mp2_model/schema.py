"""Input-only candidate schema using the unchanged official numeric parser."""
from dataclasses import dataclass
import hashlib
from official.common import parse_answer

@dataclass(frozen=True)
class Question:
    id: str
    question: str
    texts: tuple
    answers: tuple

    @classmethod
    def from_bank(cls,row):
        if [c['index'] for c in row['candidates']] != [0,1,2,3]:
            raise ValueError('Exactly four ordered candidates required')
        texts=tuple(c['text'] for c in row['candidates'])
        return cls(row['id'],row['question'],texts,tuple(parse_answer(t) for t in texts))

    @property
    def disagreement(self): return len({a for a in self.answers if a is not None})>=2
    @property
    def question_key(self): return hashlib.sha256(' '.join(self.question.split()).encode()).hexdigest()

def partition(questions,k,seed):
    """Original label-free question grouping/balancing algorithm, unchanged."""
    import numpy as np
    groups={}
    for q in questions: groups.setdefault(q.question_key,[]).append(q.id)
    if len(groups)<k: raise ValueError('Not enough question groups')
    keys=sorted(groups);order=np.random.default_rng(seed).permutation(len(keys));buckets=[[] for _ in range(k)]
    for index in order:
        bucket=min(range(k),key=lambda i:(len(buckets[i]),i))
        buckets[bucket].extend(sorted(groups[keys[index]]))
    return [sorted(b) for b in buckets]
